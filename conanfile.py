from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
import os


class opencvRecipe(ConanFile):
    name = "opencv"
    version = "0.0.1"
    package_type = "application"

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of zcc package here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*"

    def requirements(self):
        # https://docs.conan.io/2/tutorial/versioning/version_ranges.html
        # https://conan.io/center/recipes/doctest?version=2.4.11
        self.requires("doctest/[^2.4]")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        db_path = os.path.join(
            "build", str(self.settings.build_type), "compile_commands.json")
        if os.path.exists("../../compile_commands.json"):
            os.unlink("../../compile_commands.json")
        os.symlink(db_path, "../../compile_commands.json")

    def package(self):
        cmake = CMake(self)
        cmake.install()
