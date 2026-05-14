# Prompt: API Contract Design (Design-First)

> Use este prompt ANTES de implementar qualquer endpoint.
> Design-first: o contrato OpenAPI é gerado antes do código.
> Referência: [.github/copilot/skills/api-design.md], [OpenAPI 3.1 Spec](https://spec.openapis.org/oas/v3.1.0)

---

## Contexto Necessário

```
@workspace #file:.github/copilot/prompts/new-project/06-api-contract-design.md
           #file:.github/copilot/skills/api-design.md
           #file:.github/copilot/prompts/new-project/02-domain-modeling.md
           #file:[domain-model-document]
```

---

## Prompt

```
Aja como um API Designer sênior e produza o contrato OpenAPI 3.1 completo para o recurso descrito.
Siga o checklist de design do skill api-design.md em cada decisão.

## Recurso a Especificar

Recurso: [nome do recurso, ex: Orders]
Bounded Context: [contexto do DDD]
Consumers: [quem vai consumir esta API: frontend web, app mobile, serviço parceiro]
Autenticação: [Bearer JWT | API Key | OAuth2 | Sem autenticação]

## Use Cases / Operações Necessárias

Liste os casos de uso que esta API deve suportar:
  1. [caso de uso 1: ex: "Cliente visualiza seus pedidos paginados"]
  2. [caso de uso 2: ex: "Operador confirma um pedido específico"]
  3. [caso de uso 3: ex: "Sistema cria pedido a partir do carrinho"]
  4. [...]

## Dados do Domínio

[Cole aqui o Domain Model do Passo 2, ou descreva as entidades e seus campos]

## Regras de Negócio Relevantes para a API

  - [regra 1: ex: "Pedido só pode ser confirmado se estiver em status DRAFT"]
  - [regra 2: ex: "Apenas o dono do pedido pode cancelá-lo"]
  - [regra 3: ex: "Listagem retorna apenas pedidos do usuário autenticado"]

---

## Entregáveis

### 1. Decisões de Design (justificadas)

Para cada decisão importante, documente:

**URL Structure:**
  - URL base: `/v1/[recurso-no-plural]`
  - Sub-recursos: [listar relacionamentos e como foram modelados]
  - Ações não-CRUD: `POST /[recurso]/{id}/[acao]`

**Paginação:**
  - Tipo: [cursor-based | offset-based] — justificativa: [...]
  - Campos: [page/page_size | cursor/limit]

**Filtragem:**
  - Query params suportados: [lista]

**Versionamento:**
  - Estratégia: [URL path /v1/ | header API-Version]

### 2. Mapeamento de Status HTTP

| Caso | Método | Status | Quando |
|------|--------|--------|--------|
| Recurso criado | POST | 201 | [condição] |
| Recurso não encontrado | GET | 404 | [condição] |
| Regra de negócio violada | POST/PATCH | 422 | [condição] |
| Sem permissão | GET/POST | 403 | [condição] |
| [outros casos] | | | |

### 3. Especificação OpenAPI 3.1

Gere o arquivo `openapi.yaml` completo incluindo:

```yaml
openapi: "3.1.0"
info:
  title: [NomeDoServiço] API
  version: "1.0.0"
  description: |
    [Descrição do serviço]
    
    ## Autenticação
    [Descrever mecanismo]
    
    ## Rate Limiting
    [Descrever limites e headers]
    
    ## Erros
    Erros seguem o formato RFC 7807 (Problem Details).

servers:
  - url: https://api.[dominio].com/v1
    description: Production
  - url: https://staging-api.[dominio].com/v1
    description: Staging

paths:
  /[recurso]:
    get:
      operationId: list[Recurso]s
      summary: [descrição]
      tags: [[Recurso]s]
      security:
        - bearerAuth: []
      parameters:
        [query params de filtro e paginação]
      responses:
        "200":
          description: Lista paginada de [recurso]s
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/[Recurso]ListResponse"
        "401":
          $ref: "#/components/responses/Unauthorized"

    post:
      operationId: create[Recurso]
      summary: [descrição]
      [...]

  /[recurso]/{id}:
    get:
      [...]
    patch:
      [...]
    delete:
      [...]

  /[recurso]/{id}/[acao]:
    post:
      operationId: [acao][Recurso]
      summary: [descrição da ação de negócio]
      [...]

components:
  schemas:
    [Recurso]:
      type: object
      required: [id, status, created_at]
      properties:
        id:
          type: string
          format: uuid
          example: "550e8400-e29b-41d4-a716-446655440000"
        [campos com tipos, formatos e exemplos]
        created_at:
          type: string
          format: date-time
          example: "2024-05-14T10:30:00Z"

    Create[Recurso]Request:
      type: object
      required: [[campos obrigatórios]]
      properties:
        [campos do request com validações]

    [Recurso]ListResponse:
      type: object
      required: [data, pagination]
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/[Recurso]"
        pagination:
          $ref: "#/components/schemas/Pagination"

    Pagination:
      type: object
      required: [page, page_size, total, total_pages]
      properties:
        page: { type: integer, minimum: 1 }
        page_size: { type: integer, minimum: 1, maximum: 100 }
        total: { type: integer, minimum: 0 }
        total_pages: { type: integer, minimum: 1 }
        next: { type: string, nullable: true }
        prev: { type: string, nullable: true }

    ProblemDetail:
      type: object
      properties:
        type: { type: string, format: uri }
        title: { type: string }
        status: { type: integer }
        detail: { type: string }
        instance: { type: string }
        trace_id: { type: string }

  responses:
    Unauthorized:
      description: Token ausente ou inválido
      content:
        application/problem+json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"
    Forbidden:
      description: Sem permissão para este recurso
      content:
        application/problem+json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"
    NotFound:
      description: Recurso não encontrado
      content:
        application/problem+json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"
    UnprocessableEntity:
      description: Regra de negócio violada
      content:
        application/problem+json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

### 4. Exemplos de Request/Response

Para cada endpoint, forneça exemplos realistas:

```
### GET /[recurso]?page=1&page_size=20

Response 200:
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      [campos com valores de exemplo]
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 42,
    "total_pages": 3,
    "next": "/[recurso]?page=2&page_size=20",
    "prev": null
  }
}

### POST /[recurso]/{id}/[acao] — Erro 422

Response 422:
{
  "type": "https://api.[dominio].com/errors/[tipo-do-erro]",
  "title": "[Título legível do erro]",
  "status": 422,
  "detail": "[Descrição específica do motivo]",
  "instance": "/[recurso]/{id}/[acao]",
  "trace_id": "abc-def-123"
}
```

### 5. Checklist de Design

  - [ ] URLs com substantivos no plural, sem verbos
  - [ ] Ações não-CRUD como POST /{id}/{ação}
  - [ ] Status 422 (não 400) para violação de regra de negócio
  - [ ] Formato RFC 7807 em todos os erros 4xx/5xx
  - [ ] Paginação em todas as listagens
  - [ ] IDs como UUID (não sequenciais)
  - [ ] Timestamps em ISO 8601 UTC
  - [ ] Valores monetários em centavos (não float)
  - [ ] Idempotency-Key documentado para POSTs críticos
  - [ ] OpenAPI spec validada com swagger-cli ou similar

---

## Próximos Passos Após o Contrato

1. Validar o YAML: `npx @apidevtools/swagger-cli validate openapi.yaml`
2. Gerar mock server: `npx @stoplight/prism-cli mock openapi.yaml`
3. Compartilhar com consumers para feedback antes de implementar
4. Usar o contrato como input para o Agent 04 (Developer)
```

---

## Quando Usar Este Prompt

```
✅ Antes de qualquer implementação de endpoint novo
✅ Ao expor APIs para parceiros externos
✅ Ao criar contratos entre times (backend ↔ frontend)
✅ Durante refatoração de APIs existentes para nova versão

❌ Não pular mesmo em MVPs (o contrato evita retrabalho posterior)
```
