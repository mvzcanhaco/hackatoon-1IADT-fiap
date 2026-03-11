# Perfil de Linguagem: C

> **Referências Canônicas**:
> - [MISRA C:2012](https://misra.org.uk/product/misra-c2012/) — Safety-critical C guidelines
> - [NASA JPL C Coding Standard](https://lars-lab.jpl.nasa.gov/JPL_Coding_Standard_C.pdf) — Power of Ten Rules
> - [Linux Kernel Coding Style](https://www.kernel.org/doc/html/latest/process/coding-style.html)
> - [C11 Standard](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) — ISO/IEC 9899:2011
> - [SEI CERT C Coding Standard](https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+C+Coding+Standard)

---

## Detecção Automática

Copilot ativa este perfil quando o workspace contém:
- Arquivos `.c`, `.h`
- `CMakeLists.txt`, `Makefile`, `*.mk`
- Ausência de `.cpp`, `.cc` (senão usa perfil C++)

---

## Verificação do Padrão

Leia o `CMakeLists.txt` ou `Makefile` antes de gerar código:
```cmake
# Verifique a versão do padrão C em uso
set(CMAKE_C_STANDARD 11)    # C11 é o mínimo recomendado
set(CMAKE_C_STANDARD 17)    # C17 = refinamento do C11 (preferível)
```

---

## As 10 Regras de Ouro (NASA JPL — Power of Ten)

Estas regras são **obrigatórias** para código crítico e **altamente recomendadas** para qualquer código C:

```
1. COMPLEXIDADE SIMPLES: Apenas construções de fluxo simples.
   Sem recursão. Sem goto. Loops com limite de iterações verificável.

2. LIMITE DE TAMANHO: Funções não excedem 60 linhas (equivale a uma tela).

3. SEM ALOCAÇÃO DINÂMICA após inicialização:
   Malloc/free apenas na fase de setup. Zero em runtime crítico.

4. SEM CHAMADAS DE FUNÇÃO PROFUNDAS:
   Profundidade de chamada ≤ 2 níveis (exceto chamadas de sistema).

5. ASSERÇÕES EM ABUNDÂNCIA:
   Mínimo de 2 asserções por função para verificar invariantes.

6. ESCOPO MÍNIMO:
   Variáveis declaradas no menor escopo possível.
   Sem variáveis globais mutáveis.

7. VERIFICAÇÃO DE RETORNO:
   Todo valor de retorno de função não-void deve ser verificado.

8. PREPROCESSADOR LIMITADO:
   Apenas #include e #define simples. Sem macros com efeitos colaterais.

9. PONTEIROS RESTRITOS:
   Máximo 1 nível de indireção em declarações de dados.
   Nunca: void *, ponteiro para função como variável de estado.

10. CÓDIGO COMPILÁVEL SEM WARNINGS:
    -Wall -Wextra -Werror. Sem warnings suprimidos.
```

---

## Convenções de Nomenclatura

```c
/* Módulo/arquivo: snake_case */
/* Arquivo: weapon_detector.c / weapon_detector.h */

/* Tipos customizados: PascalCase com _t (convenção POSIX-safe: sem sufixo _t
   para tipos definidos pelo usuário em projetos privados) */
typedef struct Detection Detection;
typedef struct VideoFrame VideoFrame;

/* Macros e constantes: UPPER_SNAKE_CASE */
#define MAX_DETECTIONS      100
#define DEFAULT_THRESHOLD   0.7f
#define VIDEO_BUFFER_SIZE   (1024 * 1024)  /* parênteses em macros SEMPRE */

/* Funções: prefixo_do_modulo_verbo_substantivo */
int detection_create(Detection *out, const char *label, float confidence);
void detection_destroy(Detection *det);
bool detection_is_valid(const Detection *det);

/* Variáveis locais: snake_case curto mas descritivo */
int frame_count = 0;
float max_confidence = 0.0f;

/* Variáveis globais (evitar!): prefixo g_ */
static int g_initialized = 0;  /* static limita ao arquivo — preferível a global */

/* Parâmetros de output: prefixo out_ ou sufixo _out */
int frame_extract(VideoFrame *out_frame, const char *video_path, int fps);
```

---

## Header Guards e Organização

```c
/* weapon_detector.h */
#ifndef WEAPON_DETECTOR_H  /* guard único baseado no nome do arquivo */
#define WEAPON_DETECTOR_H

#ifdef __cplusplus         /* compatibilidade com C++ */
extern "C" {
#endif

#include <stdint.h>        /* tipos de largura fixa — use sempre em vez de int/long */
#include <stdbool.h>       /* bool, true, false — C99+ */
#include <stddef.h>        /* size_t, NULL, ptrdiff_t */

/* Constantes de configuração */
#define DETECTOR_MAX_LABELS     64
#define DETECTOR_LABEL_MAX_LEN  128

/* Tipos opacos (Information Hiding) */
typedef struct Detector Detector;

/* Códigos de erro: enum documentado */
typedef enum {
    DETECTOR_OK             = 0,  /* sucesso */
    DETECTOR_ERR_NULL_PTR   = -1, /* ponteiro nulo recebido */
    DETECTOR_ERR_INVALID    = -2, /* parâmetro inválido */
    DETECTOR_ERR_NO_MEMORY  = -3, /* falha de alocação */
    DETECTOR_ERR_IO         = -4, /* erro de I/O */
} DetectorError;

/* Estruturas públicas */
typedef struct {
    char    label[DETECTOR_LABEL_MAX_LEN]; /* buffer fixo — sem malloc na struct */
    float   confidence;
    uint32_t x, y, width, height;          /* tipos de largura fixa */
} Detection;

/* Assinaturas públicas documentadas */

/**
 * @brief Cria e inicializa um novo detector.
 * @param[out] out_detector Ponteiro para receber o detector criado.
 *                          Não pode ser NULL.
 * @param[in]  model_path   Caminho para o arquivo de modelo.
 *                          Não pode ser NULL ou vazio.
 * @return DETECTOR_OK em sucesso, código de erro negativo em falha.
 * @note O caller é responsável por chamar detector_destroy() ao terminar.
 */
DetectorError detector_create(Detector **out_detector, const char *model_path);

/**
 * @brief Libera todos os recursos do detector.
 * @param[in,out] detector Detector a ser destruído. Pode ser NULL (no-op).
 * @post *detector == NULL após retorno.
 */
void detector_destroy(Detector **detector);

#ifdef __cplusplus
}
#endif

#endif /* WEAPON_DETECTOR_H */
```

---

## Gerenciamento de Memória e Ponteiros

```c
/* ✅ Padrão RAII-like em C: adquirir → usar → liberar no mesmo escopo */
int process_video(const char *path, DetectionResult *out_result) {
    VideoCapture *cap = NULL;
    uint8_t *frame_buffer = NULL;
    int status = 0;

    /* Validação de entrada — SEMPRE */
    if (path == NULL || out_result == NULL) {
        return DETECTOR_ERR_NULL_PTR;
    }

    /* Alocação */
    cap = video_capture_open(path);
    if (cap == NULL) {
        status = DETECTOR_ERR_IO;
        goto cleanup;  /* goto APENAS para cleanup — padrão Linux kernel */
    }

    frame_buffer = malloc(VIDEO_FRAME_SIZE);
    if (frame_buffer == NULL) {
        status = DETECTOR_ERR_NO_MEMORY;
        goto cleanup;
    }

    /* Uso */
    status = do_processing(cap, frame_buffer, out_result);

cleanup:
    /* Liberação em ordem reversa da aquisição */
    free(frame_buffer);          /* free(NULL) é seguro — sem if necessário */
    video_capture_close(cap);    /* implementar como no-op se NULL */
    return status;
}

/* ✅ Zerar ponteiro após free — evitar use-after-free */
#define SAFE_FREE(ptr) do { free(ptr); (ptr) = NULL; } while (0)

/* ✅ Alocação de arrays — evitar overflow de multiplicação */
/* malloc(n * size) pode fazer overflow silencioso quando n é grande */
/* ❌ Proibido: */
uint8_t *buf = malloc(n * sizeof(uint8_t));  /* overflow se n > SIZE_MAX */

/* ✅ Correto: calloc já protege contra overflow e zera a memória */
uint8_t *buf = calloc(n, sizeof(uint8_t));

/* ✅ Para realloc: use reallocarray (POSIX, Linux glibc >= 2.26) */
uint8_t *new_buf = reallocarray(buf, n, sizeof(uint8_t));

/* ✅ Tipos de largura fixa para dados com tamanho definido */
uint8_t  byte_value;    /* 0 a 255 — não use unsigned char */
int32_t  pixel_count;   /* -2^31 a 2^31-1 — não use int */
uint64_t timestamp_ns;  /* timestamp em nanosegundos */
size_t   buffer_size;   /* tamanho de memória — use sempre size_t */
ptrdiff_t offset;       /* diferença entre ponteiros */
```

---

## Tratamento de Erros

```c
/* ✅ Sempre verificar retorno de funções */
int result = detector_init(&detector, config);
if (result != DETECTOR_OK) {
    fprintf(stderr, "detector_init failed: %d\n", result);
    return result;
}

/* ✅ Asserções para invariantes internas (não para input externo) */
#include <assert.h>

static void internal_process(const uint8_t *data, size_t len) {
    assert(data != NULL);     /* invariante: data nunca NULL aqui */
    assert(len > 0);          /* invariante: len sempre positivo aqui */
    assert(len <= MAX_SIZE);  /* invariante: len dentro do limite */
    /* ... */
}

/* ✅ Para builds de produção: substituir assert por error handling real */
#ifdef NDEBUG
  #define REQUIRE(cond) do { if (!(cond)) return DETECTOR_ERR_INVALID; } while (0)
#else
  #define REQUIRE(cond) assert(cond)
#endif
```

---

## Limites de Buffers e Strings

```c
/* ❌ NUNCA — buffer overflow clássico */
char buf[64];
strcpy(buf, user_input);   /* sem limite de tamanho */
sprintf(buf, "%s", str);   /* sem limite */
gets(input);               /* BANIDO — removido no C11 */

/* ✅ SEMPRE — funções com limite de tamanho */
char buf[64];
strncpy(buf, user_input, sizeof(buf) - 1);
buf[sizeof(buf) - 1] = '\0';  /* garantir terminação */

snprintf(buf, sizeof(buf), "%s: %.2f", label, confidence);

size_t n = fread(buffer, 1, sizeof(buffer), file);
if (n == 0 && ferror(file)) { /* verificar erro */ }

/* ✅ Verificação de tamanho antes de operações */
if (strlen(input) >= sizeof(destination)) {
    return ERR_BUFFER_TOO_SMALL;
}
```

---

## Compilação e Flags Obrigatórias

```makefile
# Flags obrigatórias — sem warnings é sem erros
CFLAGS := -std=c11 \
           -Wall \
           -Wextra \
           -Werror \
           -Wshadow \
           -Wformat=2 \
           -Wconversion \
           -Wstrict-prototypes \
           -pedantic

# Debug
CFLAGS_DEBUG := $(CFLAGS) -g3 -fsanitize=address,undefined -fno-omit-frame-pointer

# Release
CFLAGS_RELEASE := $(CFLAGS) -O2 -DNDEBUG
```

---

## Testes em C

```c
/* ✅ Framework de teste minimalista — Unity ou CUnit */
#include "unity.h"
#include "weapon_detector.h"

void setUp(void) { /* chamado antes de cada teste */ }
void tearDown(void) { /* chamado após cada teste */ }

void test_detector_create_returns_ok_with_valid_path(void) {
    Detector *detector = NULL;
    DetectorError result = detector_create(&detector, "models/test_model.bin");

    TEST_ASSERT_EQUAL(DETECTOR_OK, result);
    TEST_ASSERT_NOT_NULL(detector);

    detector_destroy(&detector);
    TEST_ASSERT_NULL(detector);  /* verifica que foi zerado */
}

void test_detector_create_returns_null_ptr_error_for_null_path(void) {
    Detector *detector = NULL;
    DetectorError result = detector_create(&detector, NULL);

    TEST_ASSERT_EQUAL(DETECTOR_ERR_NULL_PTR, result);
    TEST_ASSERT_NULL(detector);  /* não deve ter alocado nada */
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_detector_create_returns_ok_with_valid_path);
    RUN_TEST(test_detector_create_returns_null_ptr_error_for_null_path);
    return UNITY_END();
}
```

---

## Checklist C

- [ ] Header guard em todos os `.h`
- [ ] `extern "C"` para compatibilidade com C++
- [ ] Tipos de largura fixa (`uint8_t`, `int32_t`, `size_t`)
- [ ] Todo ponteiro de output validado como não-NULL
- [ ] `goto cleanup` para gerenciamento de recursos (padrão Linux kernel)
- [ ] `SAFE_FREE` — ponteiro zerado após `free()`
- [ ] Sem `strcpy`/`sprintf`/`gets` — use variantes `n`
- [ ] Todos os retornos de função verificados
- [ ] `assert()` para invariantes internas
- [ ] Compilação com `-Wall -Wextra -Werror` sem warnings
- [ ] Análise estática: `clang-tidy` ou `cppcheck`
