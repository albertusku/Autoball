// UnixDgramReceiver.hpp
#pragma once
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <functional>
#include <string>
#include <cstring>
#include <mutex>
#include <optional>

class UnixDgramReceiver {
public:
    using Callback = std::function<void(const std::string&)>;

    explicit UnixDgramReceiver(std::string path, int bufsize = 512)
        : path_(std::move(path)), bufsize_(bufsize) {}

    ~UnixDgramReceiver() { stop(); }

    bool start(int recv_timeout_ms = 100, Callback cb = nullptr) {
        if (running_) return true;

        // Crear socket
        fd_ = ::socket(AF_UNIX, SOCK_DGRAM, 0);
        if (fd_ < 0) { perr_("socket"); return false; }

        // Preparar dirección
        ::unlink(path_.c_str());
        sockaddr_un addr{}; addr.sun_family = AF_UNIX;
        if (path_.size() >= sizeof(addr.sun_path)) { perr_custom_("sun_path too long"); ::close(fd_); fd_=-1; return false; }
        std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path)-1);

        // bind
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == -1) {
            perr_("bind"); ::close(fd_); fd_ = -1; return false;
        }

        ::chmod(path_.c_str(), 0666); // permisos amplios (ajusta si hace falta)

        // timeout (opcional)
        if (recv_timeout_ms > 0) {
            timeval tv{}; tv.tv_sec = recv_timeout_ms / 1000; tv.tv_usec = (recv_timeout_ms % 1000) * 1000;
            if (::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == -1) perr_("setsockopt(SO_RCVTIMEO)");
        }

        cb_ = std::move(cb);
        running_ = true;
        th_ = std::thread([this]{ this->loop_(); });
        return true;
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }
        if (th_.joinable()) th_.join();
        ::unlink(path_.c_str());
    }

    // Último mensaje recibido (si hay)
    std::optional<std::string> latest() const {
        std::lock_guard<std::mutex> lk(m_);
        return last_;
    }

private:
    void loop_() {
        std::string buf; buf.resize(bufsize_);
        while (running_) {
            ssize_t n = ::recv(fd_, buf.data(), bufsize_ - 1, 0);
            if (n < 0) {
                // timeout u otro error; continuamos para permitir parada limpia
                continue;
            }
            if (n == 0) continue;
            buf[n] = '\0';
            {
                std::lock_guard<std::mutex> lk(m_);
                last_ = std::string(buf.data(), n);
            }
            if (cb_) cb_(last_.value());
        }
    }

    static void perr_(const char* where) { perror(where); }
    static void perr_custom_(const char* msg){ ::write(2, msg, std::strlen(msg)); ::write(2, "\n", 1); }

    std::string path_;
    int bufsize_{512};
    int fd_{-1};
    std::atomic<bool> running_{false};
    std::thread th_;
    Callback cb_{};
    mutable std::mutex m_;
    std::optional<std::string> last_;
};
