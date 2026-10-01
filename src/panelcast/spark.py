"""SparkSession factory that works on Databricks, Linux/macOS, and native Windows.

Databricks: reuse the platform session (serverless or classic).

Linux/macOS: the standard ``configure_spark_with_delta_pip`` route (``spark.jars.packages``).

Windows: Hadoop's local filesystem shells out to ``winutils.exe`` for chmod. Rather than
downloading third-party Windows binaries, we
  1. put the Delta jars on the driver classpath (``--packages`` triggers ``addFile`` -> chmod),
  2. swap in GlobalMentor's BareLocalFileSystem (pure Java NIO, Apache-2.0, Maven Central),
  3. point Delta's log store and Structured Streaming's checkpoint manager at FileSystem-API
     implementations so nothing goes through Hadoop's FileContext (which hard-codes the
     winutils-dependent RawLocalFileSystem).
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import requests
from pyspark.sql import SparkSession

from panelcast.paths import project_root

log = logging.getLogger(__name__)

MAVEN = "https://repo1.maven.org/maven2"
BARE_FS = ("com.globalmentor", "hadoop-bare-naked-local-fs", "0.1.0")


def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _ensure_java() -> None:
    """Find a JDK when this process started before Java was installed (stale PATH / JAVA_HOME)."""
    if os.environ.get("JAVA_HOME") or shutil.which("java"):
        return
    if os.name != "nt":
        return
    try:
        import winreg

        key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "JAVA_HOME")
        if value and Path(value).exists():
            os.environ["JAVA_HOME"] = value.rstrip("\\")
            return
    except OSError:
        pass
    for pattern in (
        r"C:\Program Files\Microsoft\jdk-*",
        r"C:\Program Files\Eclipse Adoptium\jdk-*",
        r"C:\Program Files\Java\jdk-*",
    ):
        found = sorted(glob.glob(pattern))
        if found:
            os.environ["JAVA_HOME"] = found[-1]
            return


def _delta_artifacts() -> list[tuple[str, str, str]]:
    import pyspark

    spark_mm = ".".join(pyspark.__version__.split(".")[:2])
    delta_v = version("delta-spark")
    return [
        ("io.delta", f"delta-spark_{spark_mm}_2.13", delta_v),
        ("io.delta", "delta-storage", delta_v),
    ]


def _jars_dir() -> Path:
    env = os.environ.get("PANELCAST_JARS_DIR")
    if env:
        return Path(env)
    root = project_root()
    return root / ".jars" if root else Path.home() / ".cache" / "panelcast" / "jars"


def ensure_jars(artifacts: list[tuple[str, str, str]], dest: Path | None = None) -> list[Path]:
    """Download Maven Central jars once (SHA-1 verified) and return their local paths."""
    dest = dest or _jars_dir()
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for group, artifact, ver in artifacts:
        jar = dest / f"{artifact}-{ver}.jar"
        if not jar.exists():
            url = f"{MAVEN}/{group.replace('.', '/')}/{artifact}/{ver}/{artifact}-{ver}.jar"
            log.info("downloading %s", url)
            body = requests.get(url, timeout=120).content
            expected = requests.get(url + ".sha1", timeout=30).text.split()[0].strip()
            if hashlib.sha1(body).hexdigest() != expected:  # noqa: S324 - Maven publishes SHA-1
                raise RuntimeError(f"checksum mismatch for {url}")
            jar.write_bytes(body)
        paths.append(jar)
    return paths


def get_spark(app_name: str = "panelcast", cores: int | None = None, driver_memory: str | None = None) -> SparkSession:
    """Return a Delta-enabled SparkSession for the current platform."""
    if is_databricks():
        return SparkSession.builder.getOrCreate()

    _ensure_java()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    cores = cores or int(os.environ.get("PANELCAST_CORES", min(os.cpu_count() or 4, 12)))
    driver_memory = driver_memory or os.environ.get("PANELCAST_DRIVER_MEMORY", "8g")

    builder = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", driver_memory)
        # Hostnames with underscores (common on Windows PCs) are invalid in Spark RPC URLs.
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", str(cores * 2))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.databricks.delta.snapshotPartitions", "4")
    )
    root = project_root()
    if root and (root / "conf" / "log4j2.properties").exists():
        uri = (root / "conf" / "log4j2.properties").resolve().as_uri()
        builder = builder.config("spark.driver.extraJavaOptions", f"-Dlog4j.configurationFile={uri}")

    if os.name == "nt":
        jars = ensure_jars(_delta_artifacts() + [BARE_FS])
        builder = (
            builder.config("spark.driver.extraClassPath", os.pathsep.join(str(j) for j in jars))
            .config("spark.hadoop.fs.file.impl", "com.globalmentor.apache.hadoop.fs.BareLocalFileSystem")
            .config("spark.delta.logStore.file.impl", "io.delta.storage.LocalLogStore")
            .config(
                "spark.sql.streaming.checkpointFileManagerClass",
                "org.apache.spark.sql.execution.streaming.checkpointing.FileSystemBasedCheckpointFileManager",
            )
        )
        spark = builder.getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip

        spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def set_conf(spark: SparkSession, key: str, value: str) -> None:
    """Best-effort conf setter (serverless compute rejects many keys)."""
    try:
        spark.conf.set(key, value)
    except Exception:  # noqa: BLE001
        log.debug("could not set %s on this compute", key)
