import re
import sys
import os
import time
import json
import logging
import copy
import networkx as nx
import platform
import requests
import zipfile
from typing import Any
import jax
import jaxlib
from jaxlib import version as jaxlib_version
import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax.nn
import jax.random as jrn
import jax.scipy as jsp
import torch
import torch.multiprocessing as tmp
import torch.nn as tnn

class BaseManager:
    """
    BaseManager handles versioning and environment details for the Base language.
    """
    def __init__(self):
        self._logger = self._configure_logger()
        self.version = "1.0.0"

        # Log environment details on initialization.
        self._logger.info(f"Initializing BaseManager version {self.version}")
        self._logger.info(f"JAX version: {self.get_jax_version()}")
        self._logger.info(f"JAXlib version: {self.get_jaxlib_version()}")
        self._logger.info(f"Platform: {self.get_platform()}")
        self._logger.info(f"NumPy version: {self.get_numpy_version()}")
        self._logger.info(f"SciPy version: {self.get_scipy_version()}")
        self._logger.info(f"Torch version: {self.get_torch_version()}")
        self._logger.info(f"Python version: {self.get_py_version()}")
        self._logger.info(f"JAX devices: {self.get_jax_devices()}")
        self._logger.info(f"JAX host ID: {self.get_jax_host_id()}")

        self._logger.info("BaseManager initialized successfully.")

    def _configure_logger(self):
        logger = logging.getLogger("BaseManager")
        logger.setLevel(logging.INFO)
        # Prevent adding multiple handlers if they already exist.
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def get_logger(self):
        return self._logger

    def get_jax_version(self):
        return jax.__version__

    def get_jaxlib_version(self):
        return jaxlib_version.__version__

    def get_platform(self):
        return platform.platform()

    def get_numpy_version(self):
        return np.__version__

    def get_scipy_version(self):
        return sp.__version__

    def get_torch_version(self):
        return torch.__version__

    def get_py_version(self):
        return sys.version

    def get_jax_devices(self):
        return jax.devices()

    def get_jax_host_id(self):
        return jax.host_id()

    def on_cpu(self):
        return jax.devices("cpu")

    def on_gpu(self):
        return jax.devices("gpu")

    def on_tpu(self):
        return jax.devices("tpu")

    def get_version(self):
        return self.version

    def get_device_count(self):
        return jax.device_count()

    def print_devices(self):
        self.get_logger().info(f"Devices: {jax.devices()}")

    def print_host_id(self):
        self.get_logger().info(f"Host ID: {jax.host_id()}")

    def print_device_count(self):
        self.get_logger().info(f"Device count: {jax.device_count()}")

    def print_platform(self):
        self.get_logger().info(f"Platform: {platform.platform()}")

    def print_jax_version(self):
        self.get_logger().info(f"JAX version: {jax.__version__}")

    def print_jaxlib_version(self):
        self.get_logger().info(f"JAXlib version: {jaxlib_version.__version__}")

    def print_numpy_version(self):
        self.get_logger().info(f"NumPy version: {np.__version__}")

    def print_scipy_version(self):
        self.get_logger().info(f"SciPy version: {sp.__version__}")

    def print_torch_version(self):
        self.get_logger().info(f"Torch version: {torch.__version__}")

    def print_py_version(self):
        self.get_logger().info(f"Python version: {sys.version}")

    def print_version(self):
        self.get_logger().info(f"Version: {self.version}")

class BasePackageManager:
    """
    BasePackageManager handles installation, uninstallation, and dependency resolution
    for packages used in the Base language.
    """
    def __init__(self):
        self.installed_packages = {}
        self.dependency_graph = nx.DiGraph()
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger("BasePackageManager")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def install_package(self, package_name: str, repo_url: str):
        self.logger.info(f"Installing package: {package_name}")
        package_info = self._fetch_package_info(repo_url)
        if package_info:
            self._download_and_install(package_info)
            self._resolve_dependencies(package_info)
        else:
            self.logger.error(f"Package {package_name} not found at {repo_url}.")

    def uninstall_package(self, package_name: str):
        self.logger.info(f"Uninstalling package: {package_name}")
        if package_name in self.installed_packages:
            del self.installed_packages[package_name]
            if self.dependency_graph.has_node(package_name):
                self.dependency_graph.remove_node(package_name)
            self.logger.info(f"Package {package_name} uninstalled successfully.")
        else:
            self.logger.error(f"Package {package_name} is not installed.")

    def list_installed_packages(self):
        self.logger.info("Installed packages:")
        for pkg_name, pkg_info in self.installed_packages.items():
            version = pkg_info.get('version', 'unknown')
            self.logger.info(f"{pkg_name} - {version}")

    def _fetch_package_info(self, repo_url: str) -> Any:
        package_info_url = f"{repo_url}/package.json"
        self.logger.info(f"Fetching package info from {package_info_url}")
        try:
            response = requests.get(package_info_url)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to fetch package info from {repo_url}, status code: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching package info: {e}")
            return None

    def _download_and_install(self, package_info: dict):
        download_url = package_info.get('download_url')
        package_name = package_info.get('name')
        package_version = package_info.get('version')
        if not download_url or not package_name or not package_version:
            self.logger.error("Incomplete package information; cannot proceed with installation.")
            return

        self.logger.info(f"Downloading package from {download_url}")
        try:
            response = requests.get(download_url)
            if response.status_code == 200:
                # Save the downloaded package to a temporary file
                package_file = f"/tmp/{package_name}-{package_version}.zip"
                with open(package_file, 'wb') as f:
                    f.write(response.content)
                # Extract the package
                self._extract_package(package_file, package_name)
                self.installed_packages[package_name] = package_info
                self.logger.info(f"Package {package_name} installed successfully.")
                # Optionally, clean up the temporary file
                if os.path.exists(package_file):
                    os.remove(package_file)
            else:
                self.logger.error(f"Failed to download package from {download_url}, status code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error downloading package: {e}")

    def _extract_package(self, package_file: str, package_name: str):
        self.logger.info(f"Extracting package {package_name}")
        try:
            # Ensure the target directory exists; this might require elevated privileges
            target_dir = f"/usr/local/lib/base_packages/{package_name}"
            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(package_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            self.logger.info(f"Package {package_name} extracted to {target_dir}")
        except Exception as e:
            self.logger.error(f"Error extracting package: {e}")

    def _resolve_dependencies(self, package_info: dict):
        dependencies = package_info.get('dependencies', [])
        for dependency in dependencies:
            dep_name = dependency.get('name')
            dep_version = dependency.get('version')
            if not dep_name or not dep_version:
                self.logger.error("Dependency information is incomplete; skipping.")
                continue

            if dep_name not in self.installed_packages:
                self.logger.info(f"Resolving dependency: {dep_name} {dep_version}")
                # Adjust the repo URL as needed; here it’s assumed to follow a GitHub pattern.
                dep_repo_url = f"https://github.com/user/{dep_name}"
                self.install_package(dep_name, dep_repo_url)
            # Add an edge to the dependency graph (from the current package to its dependency)
            self.dependency_graph.add_edge(package_info.get('name', 'unknown'), dep_name)