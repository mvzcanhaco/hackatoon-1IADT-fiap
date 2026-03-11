# Perfil de Linguagem: C++

> **Referências Canônicas**:
> - [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) — Stroustrup & Sutter
> - [Effective Modern C++](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/) — Scott Meyers
> - [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
> - [C++20 Standard](https://isocpp.org/std/the-standard)
> - [clang-tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/list.html)
> - [cppreference.com](https://en.cppreference.com/) — referência de API

---

## Detecção Automática

Copilot ativa este perfil quando o workspace contém:
- Arquivos `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`, `.hxx`
- `CMakeLists.txt` com `project(... CXX)`
- `conanfile.txt`, `vcpkg.json`

---

## Verificação do Padrão C++

```cmake
# Leia ANTES de gerar código
set(CMAKE_CXX_STANDARD 20)         # C++17 mínimo, C++20 preferível
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)      # sem extensões GNU — portabilidade
```

---

## Regras Fundamentais (C++ Core Guidelines)

### R — Resource Management (RAII)

```cpp
// ✅ RAII — recursos adquiridos no construtor, liberados no destrutor
class VideoCapture {
public:
    explicit VideoCapture(std::string_view path)
        : handle_{open_video(path)}  // adquire no construtor
    {
        if (!handle_) {
            throw VideoNotFoundError{std::string{path}};
        }
    }

    ~VideoCapture() {  // libera no destrutor — automático ao sair do escopo
        if (handle_) {
            close_video(handle_);
        }
    }

    // Regra dos 5 — ou Rule of Zero (preferível)
    // ⚠️ ARMADILHA: definir APENAS o destrutor desabilita silenciosamente o move!
    // O compilador deleta move ctor/assign implicitamente se o destrutor for declarado.
    // A classe cai de volta para cópia — perde performance SEM erro de compilação.
    // Solução: declare TODOS os 5 membros especiais explicitamente.
    VideoCapture(const VideoCapture&) = delete;             // não copiável
    VideoCapture& operator=(const VideoCapture&) = delete;
    VideoCapture(VideoCapture&&) noexcept = default;        // movível — DEVE ser declarado
    VideoCapture& operator=(VideoCapture&&) noexcept = default;

private:
    VideoHandle* handle_;
};

// ✅ Uso — sem delete manual
{
    VideoCapture cap{"video.mp4"};
    // usa cap...
}  // destrutor chamado automaticamente — sem leak possível

// ❌ Proibido — gerenciamento manual
VideoHandle* handle = open_video("video.mp4");
// ... esquecer de chamar close_video — leak!
```

---

### Smart Pointers — Nunca `new`/`delete` Direto

```cpp
// ✅ unique_ptr — ownership exclusivo (padrão default)
auto detector = std::make_unique<WeaponDetector>(config);

// ✅ shared_ptr — ownership compartilhado (use com cuidado — overhead)
auto model = std::make_shared<OnnxModel>("model.onnx");

// ✅ weak_ptr — referência sem ownership (quebra ciclos)
std::weak_ptr<Cache> cache_ref = shared_cache;

// ✅ Função factory com retorno por unique_ptr
std::unique_ptr<Detector> DetectorFactory::create(DeviceType device) {
    switch (device) {
        case DeviceType::GPU: return std::make_unique<GpuDetector>();
        case DeviceType::CPU: return std::make_unique<CpuDetector>();
    }
    throw std::invalid_argument{"Unknown device type"};
}

// ❌ NUNCA
Detector* d = new GpuDetector();   // ownership ambíguo
delete d;                           // manual — propenso a erro/leak
```

---

### Regra de Zero (preferível à Regra de 5)

```cpp
// ✅ Regra de Zero: deixar o compilador gerar tudo
// Use membros que gerenciam seus próprios recursos
class DetectionResult {
public:
    explicit DetectionResult(std::string video_path)
        : video_path_{std::move(video_path)}  // move para evitar cópia
    {}

    // Copiar, mover, destruir — todos gerados corretamente pelo compilador
    // porque todos os membros (string, vector, unique_ptr) gerenciam seus recursos

private:
    std::string video_path_;
    std::vector<Detection> detections_;
    std::unique_ptr<Metrics> metrics_;
};

// ⚠️ ARMADILHA — destrutor sem Regra dos 5 silencia perda de move semantics:
// class Broken {
//     ~Broken() { cleanup(); }      // declarar destrutor implicitamente...
//     // ...deleta move ctor/assign → classe vira "só cópia" SEM aviso do compilador
// };
//
// clang-tidy: cppcoreguidelines-rule-of-five detecta esta violação (C.21)
```

---

## Convenções de Nomenclatura (C++ Core Guidelines + Google)

```cpp
// Tipos (classes, structs, enums, type aliases): PascalCase
class WeaponDetector { ... };
struct DetectionConfig { ... };
enum class DeviceType { CPU, GPU };
using DetectionList = std::vector<Detection>;

// Funções e métodos: camelCase (Google) ou snake_case (projeto)
// IMPORTANTE: verifique o estilo do projeto existente antes de escolher
void processVideo(std::string_view path);  // Google style
void process_video(std::string_view path); // snake_case style

// Variáveis membros: trailing underscore_ (Google) ou m_ prefix
class Detector {
private:
    std::string model_path_;   // Google: trailing _
    int threshold_;
    // OU:
    std::string m_modelPath;   // alternativa: m_ prefix
};

// Constantes e enumeradores: kPascalCase (Google) ou UPPER_SNAKE
constexpr int kMaxFrames = 300;
constexpr float kDefaultThreshold = 0.7f;

// Arquivos: snake_case.hpp / snake_case.cpp (Google) ou PascalCase.hpp
// VERIFIQUE o padrão existente no projeto antes de criar novos
```

---

## `const` e `constexpr` em Todo Lugar

```cpp
// ✅ const por padrão — só mutable quando necessário
class VideoProcessor {
public:
    // const em parâmetros que não são modificados
    DetectionResult process(const std::string& path) const;  // método const

    // string_view em vez de const string& para strings (C++17)
    bool isValidPath(std::string_view path) const;

    // constexpr para valores conhecidos em compile time
    static constexpr int MAX_RESOLUTION = 1920;
    static constexpr float MIN_CONFIDENCE = 0.0f;
    static constexpr float MAX_CONFIDENCE = 1.0f;
};

// ✅ [[nodiscard]] para resultados que não devem ser ignorados
[[nodiscard]] std::expected<Detection, DetectorError>  // C++23
    detect(const Frame& frame) const;

// [[nodiscard]] C++17
[[nodiscard]] DetectionResult process(const std::string& path);

// Se caller ignorar o retorno: warning do compilador
process("video.mp4");  // warning: ignoring return value
```

---

## Tipos Modernos C++17/20

```cpp
#include <optional>    // C++17
#include <variant>     // C++17
#include <string_view> // C++17
#include <span>        // C++20
#include <ranges>      // C++20
#include <expected>    // C++23

// std::optional — valor ou ausência (sem nullptr)
std::optional<Detection> findByLabel(std::string_view label) const {
    auto it = std::find_if(detections_.begin(), detections_.end(),
        [&](const Detection& d) { return d.label == label; });
    if (it == detections_.end()) return std::nullopt;
    return *it;
}

// std::variant — tipo soma seguro (alternativa a union)
using ProcessingResult = std::variant<DetectionResult, DetectorError>;

ProcessingResult process(const std::string& path) {
    if (!fileExists(path)) return DetectorError::FileNotFound;
    return detectObjects(path);
}

// Visitor para variant
std::visit([](auto&& result) {
    using T = std::decay_t<decltype(result)>;
    if constexpr (std::is_same_v<T, DetectionResult>) {
        handleSuccess(result);
    } else if constexpr (std::is_same_v<T, DetectorError>) {
        handleError(result);
    }
}, result);

// std::span — view não-owning sobre arrays/vetores (C++20)
void processFrames(std::span<const Frame> frames) {  // não copia
    for (const auto& frame : frames) { ... }
}

// Ranges (C++20)
auto highConfidence = detections_
    | std::views::filter([](const Detection& d) { return d.confidence > 0.8f; })
    | std::views::transform([](const Detection& d) { return d.label; });
```

---

## Tratamento de Erros

```cpp
// ✅ Exceções para erros excepcionais (não para fluxo normal)
class DetectorError : public std::runtime_error {
public:
    explicit DetectorError(std::string_view msg, int code = -1)
        : std::runtime_error{std::string{msg}}
        , code_{code}
    {}
    int code() const noexcept { return code_; }
private:
    int code_;
};

class ModelNotFoundError : public DetectorError {
public:
    explicit ModelNotFoundError(std::string_view path)
        : DetectorError{std::string{"Model not found: "} + std::string{path}}
    {}
};

// ✅ noexcept em funções que não podem lançar
void destroy() noexcept;
~Detector() noexcept;  // destrutor: sempre noexcept

// ✅ std::expected (C++23) — alternativa a exceções para erros esperados
std::expected<Detection, DetectorError> tryDetect(const Frame& frame) {
    if (!isInitialized_) {
        return std::unexpected{DetectorError{"Detector not initialized"}};
    }
    return doDetect(frame);
}

// Uso:
auto result = tryDetect(frame);
if (!result) {
    logger.error("Detection failed: {}", result.error().what());
    return;
}
processDetection(result.value());
```

---

## Interfaces e Polimorfismo

```cpp
// ✅ Interface pura (abstrata)
class DetectorInterface {
public:
    virtual ~DetectorInterface() = default;  // destrutor virtual OBRIGATÓRIO

    // Métodos virtuais puros = interface
    [[nodiscard]] virtual DetectionResult
        processVideo(std::string_view path, const DetectionConfig& config) = 0;

    virtual void cleanup() noexcept = 0;

    // Método com implementação padrão (hook opcional)
    virtual std::string_view deviceInfo() const { return "unknown"; }
};

// ✅ Implementação concreta
class GpuDetector final : public DetectorInterface {  // final evita herança acidental
public:
    explicit GpuDetector(int device_id = 0);

    [[nodiscard]] DetectionResult
        processVideo(std::string_view path, const DetectionConfig& config) override;

    void cleanup() noexcept override;
    std::string_view deviceInfo() const override;

private:
    // membros privados...
};
```

---

## CMakeLists.txt Moderno

```cmake
cmake_minimum_required(VERSION 3.25)
project(weapon_detector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Flags de warning (obrigatórias)
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wshadow -Wnon-virtual-dtor
    -Wold-style-cast -Wcast-align
    -Woverloaded-virtual
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

# Target moderno (preferível a variáveis globais)
add_library(detector_lib STATIC
    src/domain/detector.cpp
    src/infrastructure/gpu_detector.cpp
)

target_include_directories(detector_lib
    PUBLIC  include/           # público: headers da API
    PRIVATE src/               # privado: detalhes de implementação
)

target_link_libraries(detector_lib
    PUBLIC  Threads::Threads
    PRIVATE OpenCV::OpenCV
)

# Executável
add_executable(detector_app src/main.cpp)
target_link_libraries(detector_app PRIVATE detector_lib)

# Testes
enable_testing()
add_subdirectory(tests)
```

---

## Testes com Google Test

```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "weapon_detector.hpp"

// ✅ Mock de interface
class MockDetector : public DetectorInterface {
public:
    MOCK_METHOD(DetectionResult, processVideo,
        (std::string_view path, const DetectionConfig& config), (override));
    MOCK_METHOD(void, cleanup, (), (noexcept, override));
};

// ✅ Fixture para setup/teardown compartilhado
class DetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = DetectionConfig{.threshold = 0.7f, .fps = 5};
    }

    DetectionConfig config_;
    MockDetector mock_detector_;
};

TEST_F(DetectorTest, ProcessVideoReturnsDetectionsAboveThreshold) {
    // Arrange
    EXPECT_CALL(mock_detector_, processVideo("video.mp4", config_))
        .WillOnce(::testing::Return(DetectionResult{
            .detections = {Detection{.label = "knife", .confidence = 0.9f}},
        }));

    // Act
    auto result = mock_detector_.processVideo("video.mp4", config_);

    // Assert
    EXPECT_EQ(result.detections.size(), 1u);
    EXPECT_EQ(result.detections[0].label, "knife");
    EXPECT_FLOAT_EQ(result.detections[0].confidence, 0.9f);
}

TEST_F(DetectorTest, ProcessVideoWithEmptyPathThrows) {
    EXPECT_THROW(
        real_detector_.processVideo("", config_),
        DetectorError
    );
}

// Testes parametrizados
INSTANTIATE_TEST_SUITE_P(
    ThresholdTests,
    DetectorTest,
    ::testing::Values(0.5f, 0.7f, 0.9f)
);
```

---

## Checklist C++

- [ ] Padrão C++ (`CMAKE_CXX_STANDARD`) verificado no CMakeLists.txt
- [ ] Smart pointers — sem `new`/`delete` direto
- [ ] RAII — todos os recursos gerenciados por objetos com destrutor
- [ ] Regra de Zero implementada (ou Regra de 5 completa)
- [ ] `const` em todos os parâmetros e métodos que não modificam estado
- [ ] `[[nodiscard]]` em funções cujo retorno não deve ser ignorado
- [ ] Destrutor virtual em classes base polimórficas
- [ ] `override`/`final` em todos os overrides
- [ ] `noexcept` em destrutor e funções que não lançam
- [ ] `-Wall -Wextra -Wpedantic` sem warnings na compilação
- [ ] `clang-tidy` configurado com `.clang-tidy`
- [ ] Address Sanitizer habilitado em builds de debug
