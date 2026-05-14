# Perfil de Linguagem: PHP / Laravel

## Referências Canônicas

- **PHP 8.3 Manual**: https://www.php.net/manual/en/
- **Laravel 11 Docs**: https://laravel.com/docs/11.x
- **PSR-1/PSR-2/PSR-4/PSR-12**: https://www.php-fig.org/psr/
- **PHP: The Right Way**: https://phptherightway.com/

---

## Detecção Automática

Este perfil é ativado quando o workspace contém:
- `composer.json`
- `artisan` (Laravel CLI)
- Diretório `app/` com `Http/`, `Models/`

**Verificar versão obrigatoriamente**:
```bash
cat composer.json | grep '"php"'         # Ex: "^8.3"
cat composer.json | grep '"laravel/framework"'  # Ex: "^11.0"
php --version                             # Ex: PHP 8.3.6
```

---

## Versão Mínima: PHP 8.3

Features obrigatórias por versão:

| Feature | Versão | Uso |
|---------|--------|-----|
| Named arguments | 8.0+ | Clareza em chamadas complexas |
| Attributes | 8.0+ | Substituir docblock annotations |
| Enums | 8.1+ | Status, tipos, estados |
| `readonly` properties | 8.1+ | Value Objects imutáveis |
| `readonly` classes | 8.2+ | DTOs completamente imutáveis |
| Typed class constants | 8.3+ | `const string NAME = 'value'` |
| `#[\Override]` attribute | 8.3+ | Verificar override em compile time |

---

## Estrutura de Diretórios (Hexagonal em Laravel)

```
app/
├── Domain/                          ← PHP puro — zero Laravel/Eloquent
│   ├── [Context]/
│   │   ├── Entities/
│   │   │   └── Order.php            ← Entidade com comportamento
│   │   ├── ValueObjects/
│   │   │   ├── OrderId.php          ← readonly class
│   │   │   └── Money.php
│   │   ├── Repositories/            ← Interfaces (ports)
│   │   │   └── OrderRepository.php
│   │   ├── Events/                  ← Domain Events
│   │   │   └── OrderConfirmed.php
│   │   └── Exceptions/              ← Exceções de domínio
│   │       └── OrderNotFoundException.php
│
├── Application/                     ← Orquestração (pode usar Laravel DI)
│   ├── UseCases/
│   │   └── ProcessOrderUseCase.php
│   └── DTOs/
│       ├── ProcessOrderRequest.php
│       └── ProcessOrderResponse.php
│
├── Infrastructure/                  ← Implementações concretas
│   ├── Persistence/
│   │   ├── EloquentOrderRepository.php
│   │   └── OrderEloquentModel.php   ← Model Eloquent separado da entidade
│   ├── Gateways/
│   │   └── SendGridEmailGateway.php
│   └── Providers/
│       └── DomainServiceProvider.php ← Bind interface → implementation
│
└── Http/                            ← Apresentação (Controllers, Requests, Resources)
    ├── Controllers/
    │   └── OrderController.php
    ├── Requests/                    ← Form Requests (validação HTTP)
    │   └── ProcessOrderRequest.php
    └── Resources/                  ← API Resources (serialização)
        └── OrderResource.php
```

---

## Value Object com `readonly class` (PHP 8.2+)

```php
<?php

declare(strict_types=1);

namespace App\Domain\Orders\ValueObjects;

use App\Domain\Orders\Exceptions\InvalidValueException;
use Ramsey\Uuid\Uuid;

readonly class OrderId
{
    public function __construct(
        public readonly string $value,
    ) {
        if (empty(trim($value))) {
            throw new InvalidValueException('OrderId', $value, 'cannot be empty');
        }
        if (!Uuid::isValid($value)) {
            throw new InvalidValueException('OrderId', $value, 'must be a valid UUID');
        }
    }

    public static function generate(): self
    {
        return new self(Uuid::uuid4()->toString());
    }

    public static function fromString(string $value): self
    {
        return new self($value);
    }

    public function equals(self $other): bool
    {
        return $this->value === $other->value;
    }

    public function __toString(): string
    {
        return $this->value;
    }
}
```

---

## Enum de Domínio (PHP 8.1+)

```php
<?php

declare(strict_types=1);

namespace App\Domain\Orders\ValueObjects;

enum OrderStatus: string
{
    case Draft = 'draft';
    case Confirmed = 'confirmed';
    case Cancelled = 'cancelled';

    public function canTransitionTo(self $next): bool
    {
        return match ($this) {
            self::Draft => in_array($next, [self::Confirmed, self::Cancelled]),
            self::Confirmed => $next === self::Cancelled,
            self::Cancelled => false,
        };
    }

    public function label(): string
    {
        return match ($this) {
            self::Draft => 'Rascunho',
            self::Confirmed => 'Confirmado',
            self::Cancelled => 'Cancelado',
        };
    }
}
```

---

## Entidade de Domínio

```php
<?php

declare(strict_types=1);

namespace App\Domain\Orders\Entities;

use App\Domain\Orders\Events\OrderConfirmed;
use App\Domain\Orders\Exceptions\{EmptyOrderException, InvalidStateTransitionException};
use App\Domain\Orders\ValueObjects\{CustomerId, Money, OrderId, OrderStatus};
use DateTimeImmutable;

final class Order
{
    private array $domainEvents = [];
    private array $items = [];
    private OrderStatus $status;

    private function __construct(
        private readonly OrderId $id,
        private readonly CustomerId $customerId,
        private readonly DateTimeImmutable $createdAt,
    ) {
        $this->status = OrderStatus::Draft;
    }

    public static function create(CustomerId $customerId): self
    {
        return new self(
            id: OrderId::generate(),
            customerId: $customerId,
            createdAt: new DateTimeImmutable(),
        );
    }

    public function addItem(string $productId, int $quantity, Money $unitPrice): void
    {
        if ($this->status !== OrderStatus::Draft) {
            throw new InvalidStateTransitionException('addItem', $this->status);
        }
        if ($quantity <= 0) {
            throw new \InvalidArgumentException("Quantity must be positive, got: {$quantity}");
        }
        $this->items[] = compact('productId', 'quantity', 'unitPrice');
    }

    public function confirm(): void
    {
        if (!$this->status->canTransitionTo(OrderStatus::Confirmed)) {
            throw new InvalidStateTransitionException('confirm', $this->status);
        }
        if (empty($this->items)) {
            throw new EmptyOrderException($this->id);
        }

        $this->status = OrderStatus::Confirmed;
        $this->domainEvents[] = new OrderConfirmed($this->id, $this->customerId);
    }

    public function pullDomainEvents(): array
    {
        $events = $this->domainEvents;
        $this->domainEvents = [];
        return $events;
    }

    public function getId(): OrderId { return $this->id; }
    public function getStatus(): OrderStatus { return $this->status; }
    public function getItems(): array { return $this->items; }
}
```

---

## Port Interface (Repository)

```php
<?php

declare(strict_types=1);

namespace App\Domain\Orders\Repositories;

use App\Domain\Orders\Entities\Order;
use App\Domain\Orders\ValueObjects\OrderId;

interface OrderRepository
{
    public function findById(OrderId $id): ?Order;

    /** @return Order[] */
    public function findAll(int $page = 1, int $perPage = 20): array;

    public function save(Order $order): void;

    public function delete(OrderId $id): void;

    public function existsById(OrderId $id): bool;
}
```

---

## Use Case

```php
<?php

declare(strict_types=1);

namespace App\Application\UseCases;

use App\Application\DTOs\{ProcessOrderRequest, ProcessOrderResponse};
use App\Domain\Orders\Exceptions\OrderNotFoundException;
use App\Domain\Orders\Repositories\OrderRepository;
use App\Domain\Orders\ValueObjects\OrderId;
use Illuminate\Support\Facades\Log;

final readonly class ProcessOrderUseCase
{
    public function __construct(
        private OrderRepository $orderRepository,
    ) {}

    public function execute(ProcessOrderRequest $request): ProcessOrderResponse
    {
        Log::info('Processing order', ['order_id' => $request->orderId]);

        $orderId = OrderId::fromString($request->orderId);
        $order = $this->orderRepository->findById($orderId)
            ?? throw new OrderNotFoundException($orderId);

        $order->confirm();

        $this->orderRepository->save($order);

        Log::info('Order processed successfully', ['order_id' => $request->orderId]);

        return new ProcessOrderResponse(
            orderId: (string) $order->getId(),
            status: $order->getStatus()->value,
        );
    }
}
```

---

## Implementação Eloquent do Repository

```php
<?php

declare(strict_types=1);

namespace App\Infrastructure\Persistence;

use App\Domain\Orders\Entities\Order;
use App\Domain\Orders\Repositories\OrderRepository;
use App\Domain\Orders\ValueObjects\{CustomerId, OrderId, OrderStatus};

final class EloquentOrderRepository implements OrderRepository
{
    public function findById(OrderId $id): ?Order
    {
        $model = OrderModel::find((string) $id);

        return $model ? $this->toDomain($model) : null;
    }

    public function save(Order $order): void
    {
        OrderModel::updateOrCreate(
            ['id' => (string) $order->getId()],
            [
                'customer_id' => (string) $order->getCustomerId(),
                'status' => $order->getStatus()->value,
            ]
        );
    }

    private function toDomain(OrderModel $model): Order
    {
        return Order::reconstitute(  // factory method para reconstrução
            id: OrderId::fromString($model->id),
            customerId: CustomerId::fromString($model->customer_id),
            status: OrderStatus::from($model->status),
            createdAt: new \DateTimeImmutable($model->created_at),
        );
    }
}
```

---

## Service Provider (DI Container do Laravel)

```php
<?php

declare(strict_types=1);

namespace App\Infrastructure\Providers;

use App\Application\UseCases\ProcessOrderUseCase;
use App\Domain\Orders\Repositories\OrderRepository;
use App\Infrastructure\Persistence\EloquentOrderRepository;
use Illuminate\Support\ServiceProvider;

final class DomainServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        // Bind interface → implementation
        $this->app->bind(OrderRepository::class, EloquentOrderRepository::class);

        // Registrar use cases explicitamente
        $this->app->bind(ProcessOrderUseCase::class, fn ($app) =>
            new ProcessOrderUseCase(
                orderRepository: $app->make(OrderRepository::class),
            )
        );
    }
}
```

---

## Controller (Apresentação)

```php
<?php

declare(strict_types=1);

namespace App\Http\Controllers;

use App\Application\DTOs\ProcessOrderRequest;
use App\Application\UseCases\ProcessOrderUseCase;
use App\Domain\Orders\Exceptions\{DomainException, OrderNotFoundException};
use App\Http\Requests\ProcessOrderHttpRequest;
use App\Http\Resources\OrderResource;
use Illuminate\Http\{JsonResponse, Response};

final class OrderController extends Controller
{
    public function __construct(
        private readonly ProcessOrderUseCase $processOrderUseCase,
    ) {}

    public function process(string $orderId, ProcessOrderHttpRequest $request): JsonResponse
    {
        try {
            $result = $this->processOrderUseCase->execute(
                new ProcessOrderRequest(orderId: $orderId)
            );

            return response()->json(['order_id' => $result->orderId, 'status' => $result->status]);

        } catch (OrderNotFoundException $e) {
            return response()->json(['error' => $e->getMessage()], Response::HTTP_NOT_FOUND);
        } catch (DomainException $e) {
            return response()->json(['error' => $e->getMessage()], Response::HTTP_UNPROCESSABLE_ENTITY);
        }
    }
}
```

---

## Testes com Pest PHP

```php
<?php

use App\Domain\Orders\Entities\Order;
use App\Domain\Orders\Exceptions\EmptyOrderException;
use App\Domain\Orders\ValueObjects\{CustomerId, Money, OrderStatus};

// Teste de entidade de domínio — puro PHP, sem Laravel
test('confirm transitions order to confirmed status when items present', function () {
    $order = Order::create(CustomerId::generate());
    $order->addItem('product-1', 2, Money::of('10.00', 'BRL'));

    $order->confirm();

    expect($order->getStatus())->toBe(OrderStatus::Confirmed);
});

test('confirm throws when order has no items', function () {
    $order = Order::create(CustomerId::generate());

    expect(fn () => $order->confirm())->toThrow(EmptyOrderException::class);
});

// Feature test com Laravel TestCase
test('POST /api/orders/{id}/process returns 200 on success', function () {
    $this->postJson("/api/orders/{$this->order->id}/process")
        ->assertStatus(200)
        ->assertJsonStructure(['order_id', 'status']);
});

test('POST /api/orders/{id}/process returns 404 when order not found', function () {
    $this->postJson('/api/orders/nonexistent-id/process')
        ->assertStatus(404);
});
```

---

## Checklist de Qualidade

- [ ] `declare(strict_types=1)` em todos os arquivos PHP
- [ ] `Domain/` sem imports de `Illuminate\`, `Eloquent` ou `Laravel`
- [ ] Value Objects como `readonly class` com validação no construtor
- [ ] Enums PHP 8.1+ para status e tipos de domínio
- [ ] `DomainServiceProvider` registra todas as interfaces → implementações
- [ ] Model Eloquent separado da entidade de domínio
- [ ] Form Requests validam entrada HTTP (não o domínio)
- [ ] Pest PHP para testes (não PHPUnit nativo)
- [ ] `composer.json` com `phpstan` nível 8 e `pint` para formatação
- [ ] Sem `null` sem `?Type` explícito nos tipos
