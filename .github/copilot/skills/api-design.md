# Skill: Design de APIs

> Referências: [REST API Design Rulebook](https://www.oreilly.com/library/view/rest-api-design/9781449317904/), [OpenAPI 3.1 Spec](https://spec.openapis.org/oas/v3.1.0), [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines), [Zalando RESTful API Guidelines](https://opensource.zalando.com/restful-api-guidelines/)

---

## Princípios Fundamentais

```
1. URLs identificam RECURSOS (substantivos), não ações (verbos)
2. Métodos HTTP expressam a INTENÇÃO da operação
3. Códigos HTTP comunicam o RESULTADO semântico
4. Contratos são EXPLÍCITOS (OpenAPI) e versionados
5. Respostas são PREVISÍVEIS e consistentes
```

---

## Nomenclatura de Recursos

### URLs: sempre substantivos no plural

```
✅ GET    /orders              → lista pedidos
✅ POST   /orders              → cria pedido
✅ GET    /orders/{id}         → detalha pedido
✅ PUT    /orders/{id}         → substitui pedido (idempotente)
✅ PATCH  /orders/{id}         → atualiza parcialmente
✅ DELETE /orders/{id}         → remove pedido

❌ GET    /getOrders           → verbo na URL
❌ POST   /createOrder         → verbo na URL
❌ GET    /order/{id}          → singular
❌ DELETE /orders/deleteAll    → verbo na URL
```

### Recursos aninhados — somente quando há dependência forte

```
GET  /orders/{orderId}/items          → itens de um pedido
POST /orders/{orderId}/items          → adicionar item
GET  /orders/{orderId}/items/{itemId} → detalhar item

# Máximo 2 níveis de aninhamento
❌ /customers/{id}/orders/{id}/items/{id}/reviews  → profundo demais
✅ /reviews?order_item_id={id}                     → filtro em vez de aninhamento
```

### Ações que não cabem em CRUD — usar sub-recurso de ação

```
POST /orders/{id}/confirm    → confirmar pedido
POST /orders/{id}/cancel     → cancelar pedido
POST /payments/{id}/refund   → estornar pagamento
POST /users/{id}/activate    → ativar usuário
POST /videos/{id}/process    → disparar processamento
```

---

## Métodos HTTP e Semântica

| Método | Idempotente | Safe | Uso |
|--------|------------|------|-----|
| GET | ✅ | ✅ | Leitura sem efeito colateral |
| HEAD | ✅ | ✅ | Só headers (sem body) |
| OPTIONS | ✅ | ✅ | CORS preflight |
| POST | ❌ | ❌ | Criação, ações não-idempotentes |
| PUT | ✅ | ❌ | Substituição completa |
| PATCH | ❌ | ❌ | Atualização parcial |
| DELETE | ✅ | ❌ | Remoção |

---

## Códigos HTTP — Mapeamento Semântico

```
2xx — Sucesso
  200 OK              → GET, PUT, PATCH bem-sucedidos
  201 Created         → POST que criou recurso (+ Location header)
  202 Accepted        → operação assíncrona aceita (processando)
  204 No Content      → DELETE ou PATCH sem body de retorno

4xx — Erro do cliente (não tente de novo sem mudar o request)
  400 Bad Request     → payload malformado, tipo errado
  401 Unauthorized    → não autenticado (falta ou token inválido)
  403 Forbidden       → autenticado mas sem permissão
  404 Not Found       → recurso não existe (ou não quer revelar)
  409 Conflict        → conflito de estado (duplicata, versão desatualizada)
  410 Gone            → recurso existiu mas foi permanentemente removido
  422 Unprocessable   → payload válido mas viola regra de negócio
  429 Too Many Reqs   → rate limit atingido (+ Retry-After header)

5xx — Erro do servidor (pode tentar de novo com backoff)
  500 Internal Error  → erro inesperado (log obrigatório)
  502 Bad Gateway     → dependency upstream falhou
  503 Unavailable     → sobrecarregado / manutenção (+ Retry-After)
  504 Gateway Timeout → dependency demorou demais
```

---

## Formato de Resposta

### Estrutura Consistente

```json
// Sucesso — retorna o recurso diretamente
// GET /orders/123
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "confirmed",
  "customer_id": "customer-uuid",
  "total_cents": 9990,
  "currency": "BRL",
  "created_at": "2024-05-14T10:30:00Z",
  "items": [...]
}

// Lista com paginação
// GET /orders?page=2&page_size=20
{
  "data": [...],
  "pagination": {
    "page": 2,
    "page_size": 20,
    "total": 153,
    "total_pages": 8,
    "next": "/orders?page=3&page_size=20",
    "prev": "/orders?page=1&page_size=20"
  }
}

// Erro — formato RFC 7807 (Problem Details)
// 422 Unprocessable Entity
{
  "type": "https://api.example.com/errors/order-not-editable",
  "title": "Order cannot be modified",
  "status": 422,
  "detail": "Order 123 is in 'confirmed' status and cannot accept new items",
  "instance": "/orders/123/items",
  "trace_id": "abc-def-123"
}
```

### Convenções de Campos

```
✅ snake_case para nomes de campos (não camelCase em JSON)
✅ Timestamps em ISO 8601 UTC: "2024-05-14T10:30:00Z"
✅ Datas sem hora: "2024-05-14"
✅ Valores monetários em centavos (inteiro): total_cents: 9990
✅ IDs como UUID string: "550e8400-e29b-41d4-a716-446655440000"
✅ Enums como string lowercase: "confirmed", "draft"
❌ Timestamps como Unix epoch (dificulta leitura)
❌ Valores monetários como float (imprecisão)
❌ IDs sequenciais (IDOR, information disclosure)
```

---

## Paginação

### Cursor-based (recomendado para grandes volumes)

```
GET /orders?cursor=eyJpZCI6IjEyMyJ9&limit=20

Resposta:
{
  "data": [...],
  "next_cursor": "eyJpZCI6IjE0MyJ9",  // null se última página
  "has_more": true
}
```

**Vantagens sobre offset**: consistente com inserções concorrentes, performance O(1) em qualquer página.

### Offset-based (aceitável para volumes menores)

```
GET /orders?page=3&page_size=20
GET /orders?offset=40&limit=20
```

---

## Filtros, Ordenação e Projeção

```
# Filtros como query params
GET /orders?status=confirmed&customer_id=uuid&created_after=2024-01-01

# Ordenação
GET /orders?sort=created_at:desc,total_cents:asc

# Projeção de campos (evita over-fetching)
GET /orders?fields=id,status,total_cents

# Busca textual
GET /products?q=wireless+headphones
```

---

## Idempotência

POST não é idempotente por padrão. Para operações que podem ser retentadas:

```
POST /orders
Idempotency-Key: client-generated-uuid-per-request

# Servidor armazena resultado por 24h com a chave
# Segunda chamada com mesma chave retorna o resultado original (não cria duplicata)
```

---

## Versionamento de API

### URL Path (recomendado para breaking changes)

```
https://api.example.com/v1/orders
https://api.example.com/v2/orders
```

### Header (recomendado para versões menores)

```
GET /orders
API-Version: 2024-05-01
```

### Estratégia de compatibilidade

```
Regras para NÃO quebrar clientes existentes:
✅ Adicionar novos campos (backward compatible)
✅ Adicionar novos endpoints
✅ Tornar campos obrigatórios em opcionais
❌ Remover campos existentes → cria nova versão
❌ Renomear campos → cria nova versão
❌ Mudar tipo de campo → cria nova versão
❌ Mudar semântica de enum → cria nova versão
```

---

## Especificação OpenAPI

```yaml
# openapi.yaml
openapi: "3.1.0"
info:
  title: Order Service API
  version: "1.0.0"
  description: |
    API para gerenciamento de pedidos.
    
    ## Autenticação
    Bearer token JWT no header `Authorization`.
    
    ## Rate Limiting
    100 requests/minuto por token. Header `X-RateLimit-Remaining` indica saldo.

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

paths:
  /orders:
    post:
      operationId: createOrder
      summary: Criar novo pedido
      tags: [Orders]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateOrderRequest"
            example:
              customer_id: "550e8400-e29b-41d4-a716-446655440000"
              items:
                - product_id: "prod-uuid"
                  quantity: 2
      responses:
        "201":
          description: Pedido criado com sucesso
          headers:
            Location:
              schema:
                type: string
              example: "/orders/new-order-uuid"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Order"
        "400":
          $ref: "#/components/responses/BadRequest"
        "401":
          $ref: "#/components/responses/Unauthorized"
        "422":
          $ref: "#/components/responses/UnprocessableEntity"

components:
  schemas:
    Order:
      type: object
      required: [id, status, customer_id, total_cents, currency, created_at]
      properties:
        id:
          type: string
          format: uuid
        status:
          type: string
          enum: [draft, confirmed, cancelled]
        total_cents:
          type: integer
          minimum: 0
          description: Valor total em centavos
        created_at:
          type: string
          format: date-time

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
    BadRequest:
      description: Payload malformado
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

---

## Checklist de Design de API

- [ ] URLs com substantivos no plural, sem verbos
- [ ] Métodos HTTP semânticamente corretos
- [ ] Códigos HTTP precisos (422 para regra de negócio, não 400)
- [ ] Formato de erro RFC 7807 (Problem Details) em 4xx/5xx
- [ ] Paginação em todas as listagens (cursor ou offset)
- [ ] IDs como UUID (não sequenciais)
- [ ] Timestamps em ISO 8601 UTC
- [ ] Valores monetários em centavos (não float)
- [ ] Idempotency-Key documentado para POSTs críticos
- [ ] Estratégia de versionamento definida e documentada
- [ ] OpenAPI spec completa e atualizada
- [ ] Contrato gerado antes da implementação (design-first)

---

## Instruções de Uso

```
@workspace #file:.github/copilot/skills/api-design.md
#file:src/presentation/http/[controller].py
#file:.github/copilot/agents/04-developer.md

Revise o design da API de [recurso] e proponha melhorias.
Gere o OpenAPI spec para os endpoints de [recurso].
```
