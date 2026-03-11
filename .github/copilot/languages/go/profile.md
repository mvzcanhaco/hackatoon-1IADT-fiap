# Perfil de Linguagem: Go

> **Referências Canônicas**:
> - [Effective Go](https://go.dev/doc/effective_go) — guia oficial de idiomas Go
> - [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments) — padrões de code review
> - [Organizing a Go module](https://go.dev/doc/modules/layout) — **referência oficial** do Go team para layout de módulos
> - [golang-standards/project-layout](https://github.com/golang-standards/project-layout) — layout popular da comunidade (**não é padrão oficial** do Go team)
> - [Google Go Style Guide](https://google.github.io/styleguide/go/) — guia de estilo do Google
> - [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md)
> - [Go Blog](https://go.dev/blog/) — artigos oficiais

---

## Detecção Automática

Copilot ativa este perfil quando o workspace contém:
- Arquivos `.go`, `go.mod`, `go.sum`
- `package main` ou `package [nome]` nos arquivos Go

---

## Versão Go

Verifique sempre em `go.mod`:
```go
// Leia este arquivo antes de usar features
go 1.22  // nunca use features de versão superior à declarada
```

---

## Estrutura de Projeto

### Aplicação (Layout Recomendado — go.dev/doc/modules/layout)

```
meu-projeto/
├── cmd/
│   └── server/
│       └── main.go          ← ponto de entrada (minimal — delega para internal)
├── internal/                ← código privado do projeto (não importável externamente)
│   ├── domain/
│   │   ├── entity.go        ← entidades de negócio
│   │   ├── repository.go    ← interfaces de repositório
│   │   └── service.go       ← domain services
│   ├── application/
│   │   └── usecase/
│   │       └── process.go   ← casos de uso
│   ├── infrastructure/
│   │   ├── database/
│   │   └── http/
│   └── ports/               ← interfaces de entrada/saída
├── pkg/                     ← código público reutilizável por outros projetos
│   └── errors/              ← tipos de erro públicos
├── api/                     ← definições de API (OpenAPI, proto, etc.)
├── configs/                 ← arquivos de configuração
├── scripts/                 ← scripts de build e deploy
├── tests/                   ← testes de integração e e2e
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

**Regra**: código em `internal/` não pode ser importado por outros módulos Go — use para isolar a lógica de negócio.

---

## Convenções de Nomenclatura

```go
// Pacotes: snake_case curto, sem plural desnecessário
package usecase    // ✅
package use_case   // ❌ (underscore raro em pacotes)
package usecases   // ❌ (evitar plural)

// Interfaces: sufixo -er quando possível (Effective Go)
type Reader interface { Read(p []byte) (n int, err error) }
type Detector interface { Detect(ctx context.Context, frame Frame) ([]Detection, error) }
type Repository interface { FindByID(ctx context.Context, id ID) (*Entity, error) }

// Exported = PascalCase, Unexported = camelCase
type VideoProcessor struct { ... }   // exported
type internalCache struct { ... }    // unexported

func ProcessVideo(...) { ... }       // exported function
func validateThreshold(...) { ... }  // unexported function

// Constantes e variáveis
const MaxFrames = 300     // exported constant
const defaultFPS = 5      // unexported constant
var ErrVideoNotFound = errors.New("video not found")  // sentinel error
```

---

## Tratamento de Erros (Go Way)

```go
// ✅ Padrão idiomático: erros como valores, verificados imediatamente
func (r *SQLiteRepository) FindByID(ctx context.Context, id string) (*Detection, error) {
    row := r.db.QueryRowContext(ctx, "SELECT * FROM detections WHERE id = ?", id)

    var d Detection
    if err := row.Scan(&d.ID, &d.Label, &d.Confidence); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, ErrDetectionNotFound  // sentinel error — caller decide o que fazer
        }
        return nil, fmt.Errorf("findByID %s: %w", id, err)  // wrapping com %w
    }
    return &d, nil
}

// ✅ Tipos de erro customizados
type NotFoundError struct {
    Resource string
    ID       string
}

func (e *NotFoundError) Error() string {
    return fmt.Sprintf("%s with id %q not found", e.Resource, e.ID)
}

// Verificação com errors.As (para tipos) e errors.Is (para sentinels)
var notFound *NotFoundError
if errors.As(err, &notFound) {
    http.Error(w, notFound.Error(), http.StatusNotFound)
}

// ❌ Proibido
if err != nil {
    panic(err)  // nunca panic em código de produção para erros esperados
}

_ = someFunc()  // nunca ignore erros silenciosamente
```

---

## Interfaces e Composição

```go
// ✅ Interfaces pequenas e focadas (Interface Segregation)
type FrameExtractor interface {
    ExtractFrames(ctx context.Context, videoPath string, fps int) (<-chan Frame, error)
}

type ObjectDetector interface {
    Detect(ctx context.Context, frame Frame) ([]Detection, error)
}

type Notifier interface {
    Notify(ctx context.Context, result DetectionResult) error
}

// ✅ Composição de interfaces
type VideoAnalyzer interface {
    FrameExtractor
    ObjectDetector
}

// ✅ Aceite interfaces, retorne structs concretas
func NewVideoProcessor(detector ObjectDetector, notifier Notifier) *VideoProcessor {
    return &VideoProcessor{
        detector: detector,
        notifier: notifier,
    }
}
```

---

## Context e Cancelamento

```go
// ✅ Context como primeiro parâmetro sempre
func (uc *ProcessVideoUseCase) Execute(
    ctx context.Context,
    req ProcessVideoRequest,
) (ProcessVideoResponse, error) {
    // Respeite cancelamento em operações longas
    frames, err := uc.extractor.ExtractFrames(ctx, req.VideoPath, req.FPS)
    if err != nil {
        return ProcessVideoResponse{}, fmt.Errorf("extract frames: %w", err)
    }

    for frame := range frames {
        select {
        case <-ctx.Done():
            return ProcessVideoResponse{}, ctx.Err()  // retorna context.Canceled ou DeadlineExceeded
        default:
        }
        // processa frame...
    }
}

// ✅ Timeout nas chamadas externas
ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
defer cancel()  // SEMPRE defer cancel() para evitar context leak
```

---

## Concorrência

```go
// ✅ Goroutines com WaitGroup
func processBatch(ctx context.Context, frames []Frame) []Detection {
    var (
        mu         sync.Mutex
        detections []Detection
        wg         sync.WaitGroup
    )

    sem := make(chan struct{}, runtime.NumCPU())  // semáforo para limitar concorrência

    for _, frame := range frames {
        wg.Add(1)
        go func(f Frame) {
            defer wg.Done()
            sem <- struct{}{}
            defer func() { <-sem }()

            result, err := detect(ctx, f)
            if err != nil {
                return  // logue o erro — não panic
            }
            mu.Lock()
            detections = append(detections, result...)
            mu.Unlock()
        }(frame)
    }

    wg.Wait()
    return detections
}

// ✅ Channels para pipeline
func pipeline(ctx context.Context, input <-chan Frame) <-chan Detection {
    out := make(chan Detection)
    go func() {
        defer close(out)  // SEMPRE feche o channel quando o producer terminar
        for frame := range input {
            select {
            case <-ctx.Done():
                return
            case out <- processFrame(frame):
            }
        }
    }()
    return out
}
```

---

## Testes (Table-Driven)

```go
// ✅ Table-driven tests — padrão Go idiomático
func TestCalculateConfidence(t *testing.T) {
    t.Parallel()

    tests := []struct {
        name       string
        score      float64
        threshold  float64
        wantResult bool
        wantErr    error
    }{
        {
            name:       "above threshold returns true",
            score:      0.9,
            threshold:  0.7,
            wantResult: true,
        },
        {
            name:       "below threshold returns false",
            score:      0.5,
            threshold:  0.7,
            wantResult: false,
        },
        {
            name:      "negative threshold returns error",
            score:     0.9,
            threshold: -0.1,
            wantErr:   ErrInvalidThreshold,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel()
            result, err := calculateConfidence(tt.score, tt.threshold)

            if tt.wantErr != nil {
                if !errors.Is(err, tt.wantErr) {
                    t.Errorf("got error %v, want %v", err, tt.wantErr)
                }
                return
            }
            if err != nil {
                t.Fatalf("unexpected error: %v", err)
            }
            if result != tt.wantResult {
                t.Errorf("got %v, want %v", result, tt.wantResult)
            }
        })
    }
}

// ✅ Interfaces para mocking (sem biblioteca externa quando possível)
type mockDetector struct {
    detectFn func(ctx context.Context, frame Frame) ([]Detection, error)
}

func (m *mockDetector) Detect(ctx context.Context, frame Frame) ([]Detection, error) {
    return m.detectFn(ctx, frame)
}
```

---

## Configuração e DI

```go
// ✅ Configuração via struct + functional options
type Config struct {
    DatabaseURL string
    Port        int
    LogLevel    string
    MaxWorkers  int
}

type Option func(*Config)

func WithPort(port int) Option {
    return func(c *Config) { c.Port = port }
}

func NewConfig(opts ...Option) Config {
    cfg := Config{
        Port:       8080,  // defaults
        MaxWorkers: runtime.NumCPU(),
        LogLevel:   "info",
    }
    for _, opt := range opts {
        opt(&cfg)
    }
    return cfg
}

// ✅ Wire/manual DI em main.go — não magic frameworks
func main() {
    cfg := NewConfig(WithPort(9090))

    db := database.New(cfg.DatabaseURL)
    repo := repository.NewSQLiteDetectionRepository(db)
    useCase := usecase.NewProcessVideoUseCase(repo)
    handler := http.NewDetectionHandler(useCase)

    server := http.NewServer(cfg.Port, handler)
    server.ListenAndServe()
}
```

---

## Checklist Go

- [ ] `go.mod` lido para verificar versão mínima do Go
- [ ] Todos os erros verificados (sem `_ = func()`)
- [ ] `context.Context` como primeiro parâmetro
- [ ] `defer cancel()` após todo `context.WithTimeout/WithCancel`
- [ ] Interfaces definidas no pacote **consumidor** (não produtor)
- [ ] Goroutines com tratamento de lifecycle (WaitGroup ou errgroup)
- [ ] Channels fechados pelo producer, nunca pelo consumer
- [ ] Testes table-driven com `t.Parallel()`
- [ ] Erros com `%w` para wrapping (não `%s`)
- [ ] `go vet` e `golangci-lint` passando

---

## Makefile Go

```makefile
.PHONY: build test lint vet run

build:
	go build -ldflags="-s -w" -o bin/server ./cmd/server

test:
	go test -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

lint:
	golangci-lint run ./...

vet:
	go vet ./...

run:
	go run ./cmd/server

tidy:
	go mod tidy
```
