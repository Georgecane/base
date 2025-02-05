import re
import sys
import os
import time
import json
import logging
import copy
import networkx as nx
import jax
import jaxlib
from jaxlib import version as jaxlib_version
import platform
import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax.nn
import jax.random as jrn
import jax.scipy as jsp
import torch
import torch.multiprocessing as tmp
import torch.nn as tnn
import requests
import zipfile

class BaseManager:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))
        self.version = "1.0.0"

        self.get_jax_version()
        self.get_jaxlib_version()
        self.get_platform()
        self.get_numpy_version()
        self.get_scipy_version()
        self.get_torch_version()
        self.get_py_version()
        self.get_jax_devices()
        self.get_jax_host_id()
        self.get_version()

        self.get_logger().info("Base class initialized")

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
    def __init__(self):
        self.installed_packages = {}
        self.dependency_graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def install_package(self, package_name, repo_url):
        self.logger.info(f"Installing package: {package_name}")
        package_info = self._fetch_package_info(repo_url)
        if package_info:
            self._download_and_install(package_info)
            self._resolve_dependencies(package_info)
        else:
            self.logger.error(f"Package {package_name} not found.")

    def uninstall_package(self, package_name):
        self.logger.info(f"Uninstalling package: {package_name}")
        if package_name in self.installed_packages:
            del self.installed_packages[package_name]
            self.dependency_graph.remove_node(package_name)
            self.logger.info(f"Package {package_name} uninstalled successfully.")
        else:
            self.logger.error(f"Package {package_name} is not installed.")

    def list_installed_packages(self):
        self.logger.info("Installed packages:")
        for package_name, package_info in self.installed_packages.items():
            self.logger.info(f"{package_name} - {package_info['version']}")

    def _fetch_package_info(self, repo_url):
        try:
            response = requests.get(f"{repo_url}/package.json")
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to fetch package info from {repo_url}")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching package info: {e}")
            return None

    def _download_and_install(self, package_info):
        download_url = package_info['download_url']
        package_name = package_info['name']
        package_version = package_info['version']
        self.logger.info(f"Downloading package from {download_url}")
        try:
            response = requests.get(download_url)
            if response.status_code == 200:
                package_file = f"/tmp/{package_name}-{package_version}.zip"
                with open(package_file, 'wb') as f:
                    f.write(response.content)
                self._extract_package(package_file, package_name)
                self.installed_packages[package_name] = package_info
                self.logger.info(f"Package {package_name} installed successfully.")
            else:
                self.logger.error(f"Failed to download package from {download_url}")
        except Exception as e:
            self.logger.error(f"Error downloading package: {e}")

    def _extract_package(self, package_file, package_name):
        self.logger.info(f"Extracting package {package_name}")
        try:
            with zipfile.ZipFile(package_file, 'r') as zip_ref:
                zip_ref.extractall(f"/usr/local/lib/base_packages/{package_name}")
        except Exception as e:
            self.logger.error(f"Error extracting package: {e}")

    def _resolve_dependencies(self, package_info):
        dependencies = package_info.get('dependencies', [])
        for dependency in dependencies:
            dep_name = dependency['name']
            dep_version = dependency['version']
            if dep_name not in self.installed_packages:
                self.logger.info(f"Resolving dependency: {dep_name} {dep_version}")
                # Assuming the dependency is also hosted on GitHub
                dep_repo_url = f"https://github.com/user/{dep_name}"
                self.install_package(dep_name, dep_repo_url)
            self.dependency_graph.add_edge(package_info['name'], dep_name)
