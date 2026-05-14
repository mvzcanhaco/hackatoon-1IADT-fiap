# Fase 5: Release

## Objetivo

Levar o código aprovado ao ar com segurança: build da imagem, deploy controlado, smoke tests e rollback automatizado se necessário.

**Duração típica**: 30-90 minutos  
**Agente**: `07-devops`  
**Participantes**: DevOps / Tech Lead

---

## Entrada

- Merge aprovado na main (Fase 4)
- Pipeline CI verde
- Migrations revisadas e testadas

---

## Como Ativar

```
@workspace #file:.github/copilot/sdlc/phases/05-release.md
           #file:.github/copilot/agents/07-devops.md
           #file:.github/workflows/ci.yml

Configure o release para [nome da feature/versão].
Ambiente alvo: [staging / production]
Estratégia: [rolling / blue-green / canary]
```

---

## Estratégias de Deploy

### Rolling Update (padrão — baixo risco)

```
Substitui instâncias gradualmente (1 por vez):

v1 v1 v1 v1    →    v2 v1 v1 v1    →    v2 v2 v1 v1    →    v2 v2 v2 v2
```

- **Quando usar**: deploys rotineiros, sem breaking changes
- **Rollback**: re-deploy da versão anterior

### Blue-Green (zero downtime)

```
ANTES:  Load Balancer → [Blue: v1] (ativo)     [Green: v2] (standby)
DEPOIS: Load Balancer → [Blue: v1] (standby)   [Green: v2] (ativo)

Rollback instantâneo: trocar LB de volta para Blue
```

- **Quando usar**: releases de alto impacto, com mudanças de schema
- **Rollback**: trocar roteamento do load balancer (<1 min)

### Canary (gradual)

```
5% → v2  │  95% → v1      [monitorar por 30min]
      ↓ se OK
25% → v2  │  75% → v1     [monitorar por 30min]
      ↓ se OK
100% → v2                  [completo]
```

- **Quando usar**: features de alto risco, mudanças de comportamento
- **Rollback**: redirecionar 100% para v1

---

## Pipeline de Release

```yaml
# Gatilho automático após merge na main
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [staging, production]
```

### Estágios do Pipeline

```
1. Qualidade (já validado no CI do PR)
   └── Reexecutar lint + tests como safety net

2. Build
   └── docker build --target runtime
   └── tag: sha-${GITHUB_SHA::8}, :latest, :v1.2.3 (se tag)
   └── push para registry

3. Run Migrations
   └── alembic upgrade head (ou equivalente)
   └── Verificar backward compatibility com versão anterior

4. Deploy
   └── Estratégia escolhida (rolling/blue-green/canary)
   └── Health check: aguardar todos os pods ficarem healthy

5. Smoke Tests
   └── Verificar endpoints críticos (ver lista abaixo)
   └── Se falhar: rollback automático

6. Notificação
   └── Slack/Teams: deploy concluído (ou falhou + rollback)
```

---

## Checklist Pré-Deploy

```
□ Pipeline CI da main está verde
□ Migrations testadas em staging com data real
□ Feature flags configuradas (se usar)
□ Rollback testado em staging (sabe como fazer)
□ Time de on-call notificado do deploy
□ Release Notes escritas
□ Runbook atualizado se há nova operação
```

---

## Template: Smoke Test Suite

```python
# tests/smoke/test_smoke.py
# Executado imediatamente após o deploy
import httpx
import pytest

BASE_URL = os.getenv("APP_URL", "http://localhost:8000")

@pytest.mark.smoke
def test_health_endpoint_returns_ok():
    """Health check deve responder em < 2s."""
    response = httpx.get(f"{BASE_URL}/health", timeout=2.0)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.smoke
def test_critical_flow_[name]():
    """Fluxo crítico de negócio deve funcionar end-to-end."""
    # Testa apenas o caminho feliz do fluxo mais crítico
    response = httpx.post(f"{BASE_URL}/[endpoint]", json={...})
    assert response.status_code in [200, 201]

@pytest.mark.smoke
def test_database_connectivity():
    """API deve responder com dados reais do banco."""
    response = httpx.get(f"{BASE_URL}/[recurso]?limit=1")
    assert response.status_code == 200
```

---

## Rollback Procedure

```bash
# 1. Identificar a versão anterior saudável
docker images | grep $IMAGE_NAME | head -5

# 2. Rolling rollback para versão anterior
kubectl set image deployment/[app] app=$IMAGE_NAME:sha-[sha-anterior]
# ou
docker service update --image $IMAGE_NAME:sha-[sha-anterior] [service-name]

# 3. Verificar que pods ficaram healthy
kubectl rollout status deployment/[app]

# 4. Re-executar smoke tests
pytest tests/smoke/ -m smoke

# 5. Notificar time
# "Rollback concluído. Versão sha-[anterior] restaurada. Investigando causa raiz."
```

**Meta de rollback**: < 5 minutos do momento da detecção ao serviço restaurado.

---

## Template: Release Notes

```markdown
# Release v[X.Y.Z] — [Data]

## Resumo
[1-2 frases descrevendo o que foi liberado]

## Novidades
- **[Feature]**: [Descrição para o usuário, não técnica]
- **[Melhoria]**: [Descrição]

## Correções
- [Bug corrigido]: [Descrição do comportamento anterior e atual]

## Breaking Changes
- ⚠️ [Mudança que afeta integrações]: [Como migrar]

## Infraestrutura
- Migrations executadas: [lista]
- Configurações novas necessárias: [variáveis de ambiente]

## Rollback
Para reverter: `[comando de rollback]`
Versão anterior: `sha-[sha]` / `v[versão-anterior]`
```

---

## Gate: Deploy Confirmado

- [ ] Smoke tests passando (100%)
- [ ] Health check retornando 200 em todos os pods
- [ ] Métricas de erro (5xx) normais (< 0.1%)
- [ ] Latência p95 dentro do SLO definido
- [ ] Release Notes publicadas
- [ ] Time notificado

---

## Próxima Fase

→ [Fase 6: Operate](06-operate.md) — monitoramento contínuo pós-deploy.
