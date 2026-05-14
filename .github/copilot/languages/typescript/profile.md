# Perfil de Linguagem: TypeScript / Node.js

## Referências Canônicas

- **TypeScript Handbook**: https://www.typescriptlang.org/docs/handbook/
- **Node.js Best Practices**: https://github.com/goldbergyoni/nodebestpractices
- **TypeScript Deep Dive**: https://basarat.gitbook.io/typescript/
- **Google TypeScript Style Guide**: https://google.github.io/styleguide/tsguide.html

---

## Detecção Automática

Este perfil é ativado quando o workspace contém:
- `tsconfig.json`
- Arquivos `.ts` ou `.tsx`
- `package.json` com dependência `typescript`

**Verificar versão obrigatoriamente**:
```bash
cat package.json | grep '"typescript"'  # Ex: "^5.3.0"
node --version                           # Ex: v20.11.0 (LTS)
```

---

## Stack Recomendada

| Categoria | Biblioteca | Versão mínima |
|-----------|-----------|---------------|
| Runtime | Node.js | 20 LTS |
| Linguagem | TypeScript | 5.x |
| Framework HTTP | Fastify ou Express | Fastify 4+ / Express 4+ |
| Validação | Zod | 3.x |
| ORM | Prisma ou Drizzle | Prisma 5+ / Drizzle 0.29+ |
| Testes | Vitest | 1.x |
| Linter | ESLint + typescript-eslint | 7.x |
| Formatter | Prettier | 3.x |
| DI Container | tsyringe ou inversify | opcional |

---

## Configuração: tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "exactOptionalPropertyTypes": true,
    "useUnknownInCatchVariables": true,
    "allowImportingTsExtensions": false,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## Convenções de Nomenclatura

| Elemento | Convenção | Exemplo |
|----------|-----------|---------|
| Classe / Interface / Type | `PascalCase` | `OrderRepository`, `ProcessVideoUseCase` |
| Função / Método | `camelCase` | `findById()`, `processVideo()` |
| Variável / Parâmetro | `camelCase` | `orderId`, `videoPath` |
| Constante (valor fixo) | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Constante (objeto/enum) | `PascalCase` | `OrderStatus`, `HttpStatusCode` |
| Arquivo de módulo | `kebab-case` | `order-repository.ts`, `process-video.ts` |
| Arquivo de teste | `*.test.ts` | `order.test.ts`, `process-video.use-case.test.ts` |
| Interface (Port) | `*Repository`, `*Gateway`, `*Notifier` | `OrderRepository`, `EmailGateway` |
| Enum | `PascalCase` com valores `PascalCase` | `OrderStatus.Confirmed` |

---

## Type System — Regras Obrigatórias

### strict: true — Sempre ativo

```typescript
// ❌ NUNCA — tipos implícitos
function process(data) { return data.id; }

// ✅ SEMPRE — tipos explícitos
function process(data: VideoData): string { return data.id; }
```

### Branded Types para Value Objects

```typescript
// Previne confusão entre IDs de tipos diferentes
type OrderId = string & { readonly _brand: "OrderId" };
type CustomerId = string & { readonly _brand: "CustomerId" };

function createOrderId(value: string): OrderId {
  if (!value.trim()) throw new InvalidValueError("OrderId cannot be empty");
  return value as OrderId;
}

// ❌ Erro de compilação — tipos incompatíveis
const customerId: CustomerId = createOrderId("123");
```

### `satisfies` Operator (TypeScript 4.9+)

```typescript
// Valida estrutura sem perder tipo literal
const config = {
  maxRetries: 3,
  timeout: 5000,
  env: "production",
} satisfies AppConfig;

// config.env tem tipo "production", não string genérica
```

### `import type` — Obrigatório para importações de tipo

```typescript
// ✅ Importação de tipo — zero runtime overhead
import type { Order } from "../domain/entities/order.js";
import type { OrderRepository } from "../domain/ports/order-repository.js";

// ✅ Importação de valor — quando precisa do objeto em runtime
import { OrderStatus } from "../domain/value-objects/order-status.js";
```

### `unknown` em vez de `any`

```typescript
// ❌ NUNCA
function handleError(error: any) { console.log(error.message); }

// ✅ SEMPRE
function handleError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
```

---

## Tratamento de Erros

### Hierarquia de Exceções de Domínio

```typescript
// src/domain/exceptions/domain-exception.ts
export abstract class DomainException extends Error {
  abstract readonly code: string;

  constructor(message: string, readonly context?: Record<string, unknown>) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

export class EntityNotFoundError extends DomainException {
  readonly code = "ENTITY_NOT_FOUND";

  constructor(entityName: string, id: string) {
    super(`${entityName} not found: id=${id}`, { entityName, id });
  }
}

export class BusinessRuleViolationError extends DomainException {
  readonly code = "BUSINESS_RULE_VIOLATION";

  constructor(rule: string, context?: Record<string, unknown>) {
    super(`Business rule violated: ${rule}`, context);
  }
}
```

### Result Type (sem exceções para fluxos esperados)

```typescript
// Alternativa funcional para fluxos que podem falhar
type Result<T, E = DomainException> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

function ok<T>(value: T): Result<T> {
  return { ok: true, value };
}

function err<E extends DomainException>(error: E): Result<never, E> {
  return { ok: false, error };
}

// Uso
function findOrder(id: OrderId): Result<Order, EntityNotFoundError> {
  const order = store.get(id);
  if (!order) return err(new EntityNotFoundError("Order", id));
  return ok(order);
}
```

---

## Padrões de Módulo

### Estrutura de Diretórios (Hexagonal)

```
src/
├── domain/
│   ├── entities/          ← Entidades com comportamento
│   │   └── order.ts
│   ├── value-objects/     ← Value Objects imutáveis
│   │   ├── order-id.ts
│   │   └── money.ts
│   ├── ports/             ← Interfaces (Ports) — nunca implementações
│   │   ├── order-repository.ts
│   │   └── email-gateway.ts
│   ├── events/            ← Domain Events
│   │   └── order-confirmed.ts
│   └── exceptions/        ← Exceções de domínio
│       └── domain-exception.ts
├── application/
│   ├── use-cases/         ← Um arquivo por use case
│   │   └── process-order.use-case.ts
│   └── dtos/              ← DTOs (fronteira aplicação ↔ domínio)
│       └── process-order.dto.ts
├── infrastructure/
│   ├── repositories/      ← Implementações de Repository ports
│   │   └── prisma-order-repository.ts
│   ├── gateways/          ← Implementações de Gateway ports
│   │   └── sendgrid-email-gateway.ts
│   └── config/
│       └── database.ts
└── presentation/
    ├── http/
    │   ├── routes/        ← Definição de rotas
    │   │   └── order.routes.ts
    │   └── schemas/       ← Schemas Zod (validação de entrada HTTP)
    │       └── order.schema.ts
    └── dependencies.ts    ← Grafo de DI
```

### Exports com barrel index.ts — Use com moderação

```typescript
// src/domain/entities/index.ts — apenas para domínio público
export type { Order } from "./order.js";
export type { OrderStatus } from "./order-status.js";

// ❌ EVITAR barrel em infraestrutura — dificulta tree-shaking
```

---

## Validação com Zod

```typescript
import { z } from "zod";

// Schema de entrada HTTP (camada de apresentação)
export const CreateOrderSchema = z.object({
  customerId: z.string().uuid(),
  items: z.array(
    z.object({
      productId: z.string().uuid(),
      quantity: z.number().int().positive().max(100),
    })
  ).min(1),
});

export type CreateOrderInput = z.infer<typeof CreateOrderSchema>;

// Uso no controller
const result = CreateOrderSchema.safeParse(req.body);
if (!result.success) {
  return res.status(422).json({ errors: result.error.flatten() });
}
// result.data é tipado automaticamente
```

---

## Logging

```typescript
import { pino } from "pino";

// Configuração estruturada (JSON em produção, pretty em dev)
const logger = pino({
  level: process.env.LOG_LEVEL ?? "info",
  transport: process.env.NODE_ENV === "development"
    ? { target: "pino-pretty" }
    : undefined,
});

// Contexto por módulo
const log = logger.child({ module: "ProcessOrderUseCase" });

// ✅ Logging com contexto estruturado
log.info({ orderId, customerId }, "Processing order");
log.error({ error, orderId }, "Failed to process order");

// ❌ NUNCA — dados sensíveis em logs
log.info({ password, creditCard }, "User data"); // PROIBIDO
```

---

## Configuração de Linter: ESLint

```javascript
// eslint.config.mjs
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.json",
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/explicit-function-return-type": "error",
      "@typescript-eslint/consistent-type-imports": ["error", { prefer: "type-imports" }],
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "@typescript-eslint/prefer-readonly": "error",
      "no-console": "error",
    },
  },
);
```

---

## Testes com Vitest

### Configuração

```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: false,
    environment: "node",
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      thresholds: {
        branches: 80,
        functions: 85,
        lines: 85,
        statements: 85,
      },
    },
  },
});
```

### Unitário — Entidade de Domínio

```typescript
import { describe, it, expect } from "vitest";
import { Order } from "../../src/domain/entities/order.js";
import { OrderBuilder } from "../builders/order-builder.js";
import { BusinessRuleViolationError } from "../../src/domain/exceptions/domain-exception.js";

describe("Order", () => {
  describe("confirm()", () => {
    it("should transition to Confirmed when order has items", () => {
      const order = new OrderBuilder().withItem("product-1", 2).build();
      order.confirm();
      expect(order.status).toBe(OrderStatus.Confirmed);
    });

    it("should throw BusinessRuleViolationError when order is empty", () => {
      const order = new OrderBuilder().build();
      expect(() => order.confirm()).toThrow(BusinessRuleViolationError);
    });

    it("should throw when order is already confirmed", () => {
      const order = new OrderBuilder().inConfirmedState().build();
      expect(() => order.confirm()).toThrow(/already confirmed/i);
    });
  });
});
```

### Unitário — Use Case (com vi.fn())

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { OrderRepository } from "../../src/domain/ports/order-repository.js";
import { ProcessOrderUseCase } from "../../src/application/use-cases/process-order.use-case.js";
import { OrderBuilder } from "../builders/order-builder.js";

describe("ProcessOrderUseCase", () => {
  let mockRepository: OrderRepository;
  let useCase: ProcessOrderUseCase;

  beforeEach(() => {
    mockRepository = {
      findById: vi.fn(),
      save: vi.fn(),
    };
    useCase = new ProcessOrderUseCase(mockRepository);
  });

  it("should save order after processing", async () => {
    const order = new OrderBuilder().withItem("p1", 1).build();
    vi.mocked(mockRepository.findById).mockResolvedValue(order);

    await useCase.execute({ orderId: order.id });

    expect(mockRepository.save).toHaveBeenCalledOnce();
  });

  it("should throw EntityNotFoundError when order does not exist", async () => {
    vi.mocked(mockRepository.findById).mockResolvedValue(null);

    await expect(useCase.execute({ orderId: "nonexistent" }))
      .rejects.toThrow(EntityNotFoundError);
  });
});
```

---

## Checklist de Qualidade

- [ ] `strict: true` no `tsconfig.json`
- [ ] `import type` para todas importações de tipo
- [ ] Nenhum `any` no código de produção (apenas `unknown`)
- [ ] Branded Types para Value Objects com identidade
- [ ] Zod schemas na camada de apresentação (não no domínio)
- [ ] Exceções de domínio customizadas (não `Error` genérico)
- [ ] Logging estruturado com contexto (pino ou winston)
- [ ] ESLint + typescript-eslint strict configurados
- [ ] Prettier integrado ao ESLint
- [ ] Testes com Vitest, cobertura ≥ 80%
- [ ] `noUncheckedIndexedAccess: true` — nunca acessa array sem checar
- [ ] Sem `console.log` no código de produção
