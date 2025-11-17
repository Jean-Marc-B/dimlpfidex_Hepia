FROM debian:bookworm-slim
WORKDIR /tmp
RUN apt-get update && \
apt-get install -y wget build-essential python3-dev python3-pybind11 && \
wget -O install_cmake.sh https://github.com/Kitware/CMake/releases/download/v4.2.0-rc2/cmake-4.2.0-rc2-linux-x86_64.sh && \
chmod +x install_cmake.sh && \
bash install_cmake.sh --skip-license --prefix=/usr/local
WORKDIR /app
ENTRYPOINT ["bash", "docker-build.sh"]