# Perfil de Linguagem: Android / Kotlin

> **Referências Canônicas**:
> - [Android Architecture Guide](https://developer.android.com/topic/architecture) — oficial Google
> - [Kotlin Coding Conventions](https://kotlinlang.org/docs/coding-conventions.html) — oficial JetBrains
> - [Modern Android Development (MAD)](https://developer.android.com/modern-android-development) — guia Google
> - [Android App Architecture](https://developer.android.com/topic/architecture/intro) — guia oficial
> - [Compose Best Practices](https://developer.android.com/develop/ui/compose/performance/bestpractices)
> - [Coroutines Guide](https://kotlinlang.org/docs/coroutines-guide.html)

---

## Detecção Automática

Copilot ativa este perfil quando o workspace contém:
- `build.gradle.kts` ou `build.gradle`
- `AndroidManifest.xml`
- Arquivos `.kt` com `android.` imports
- `settings.gradle.kts` com `android` block

---

## Verificação de Versões

Leia ANTES de gerar qualquer código:
```kotlin
// build.gradle.kts (app)
android {
    compileSdk = 35  // leia este valor
    minSdk = 24      // código deve ser compatível com esta versão
    targetSdk = 35
}

// libs.versions.toml ou build.gradle (verificar versões reais das libs)
[versions]
compose-bom = "2024.06.00"
kotlin = "2.0.0"
hilt = "2.51.1"
```

---

## Arquitetura — MAD + Clean Architecture

### Camadas (oficial Google)

```
┌─────────────────────────────────────────────────────────┐
│                    UI Layer                             │
│  Composable / Fragment / Activity                       │
│  ViewModel (StateHolder)                                │
│  UiState (sealed class / data class)                   │
├─────────────────────────────────────────────────────────┤
│                  Domain Layer (opcional)                │
│  UseCase / Interactor                                   │
│  Domain Models                                          │
│  Repository Interfaces                                  │
├─────────────────────────────────────────────────────────┤
│                   Data Layer                            │
│  Repository Implementations                             │
│  Remote DataSource (Retrofit)                           │
│  Local DataSource (Room)                                │
│  Data Models / DTOs                                     │
└─────────────────────────────────────────────────────────┘
```

### Estrutura de Módulos

```
app/
├── src/main/
│   ├── java/com/empresa/app/
│   │   ├── di/                    ← Hilt modules
│   │   │   ├── AppModule.kt
│   │   │   ├── DatabaseModule.kt
│   │   │   └── NetworkModule.kt
│   │   ├── domain/
│   │   │   ├── model/             ← entidades de domínio (puras, sem Android)
│   │   │   ├── repository/        ← interfaces
│   │   │   └── usecase/           ← casos de uso
│   │   ├── data/
│   │   │   ├── local/
│   │   │   │   ├── dao/           ← Room DAOs
│   │   │   │   └── entity/        ← Room entities
│   │   │   ├── remote/
│   │   │   │   ├── api/           ← Retrofit interfaces
│   │   │   │   └── dto/           ← response DTOs
│   │   │   └── repository/        ← implementações
│   │   └── presentation/
│   │       ├── ui/
│   │       │   ├── screen/        ← Composables por tela
│   │       │   └── component/     ← Composables reutilizáveis
│   │       └── viewmodel/         ← ViewModels
│   └── res/
└── build.gradle.kts
```

---

## ViewModel + UiState (Padrão Oficial)

```kotlin
// ✅ UiState como sealed class ou data class
data class DetectionUiState(
    val isLoading: Boolean = false,
    val detections: List<DetectionUi> = emptyList(),
    val error: String? = null,
    val isIdle: Boolean = true,
)

// ✅ ViewModel com StateFlow (preferível a LiveData)
@HiltViewModel
class DetectionViewModel @Inject constructor(
    private val processVideoUseCase: ProcessVideoUseCase,
) : ViewModel() {

    private val _uiState = MutableStateFlow(DetectionUiState())
    val uiState: StateFlow<DetectionUiState> = _uiState.asStateFlow()

    fun processVideo(videoPath: String) {
        viewModelScope.launch {  // usa viewModelScope — cancela com o ViewModel
            _uiState.update { it.copy(isLoading = true, isIdle = false) }

            processVideoUseCase(videoPath)
                .onSuccess { result ->
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            detections = result.map { it.toUiModel() },
                        )
                    }
                }
                .onFailure { error ->
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            error = error.message ?: "Unknown error",
                        )
                    }
                }
        }
    }

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }
}
```

---

## Jetpack Compose — Boas Práticas

```kotlin
// ✅ Composable stateless (preferível — testável e reutilizável)
@Composable
fun DetectionList(
    detections: List<DetectionUi>,
    onDetectionClick: (DetectionUi) -> Unit,
    modifier: Modifier = Modifier,  // sempre parâmetro modifier no final
) {
    LazyColumn(modifier = modifier) {
        items(
            items = detections,
            key = { it.id },  // key para otimização de recomposição
        ) { detection ->
            DetectionItem(
                detection = detection,
                onClick = { onDetectionClick(detection) },
            )
        }
    }
}

// ✅ Screen stateful (apenas na raiz da tela, conecta ao ViewModel)
@Composable
fun DetectionScreen(
    viewModel: DetectionViewModel = hiltViewModel(),
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    DetectionScreenContent(
        uiState = uiState,
        onProcessVideo = viewModel::processVideo,
        onClearError = viewModel::clearError,
    )
}

// ✅ Separar conteúdo da tela para testar sem ViewModel
@Composable
internal fun DetectionScreenContent(
    uiState: DetectionUiState,
    onProcessVideo: (String) -> Unit,
    onClearError: () -> Unit,
    modifier: Modifier = Modifier,
) {
    // conteúdo da tela
}

// ✅ Estabilidade de parâmetros (evitar recomposição desnecessária)
@Stable  // ou @Immutable para objetos imutáveis
data class DetectionUi(
    val id: String,
    val label: String,
    val confidence: Float,
)
```

---

## Repository Pattern

```kotlin
// ✅ Interface no domínio
interface DetectionRepository {
    suspend fun getAll(): Result<List<Detection>>
    suspend fun findById(id: String): Result<Detection?>
    suspend fun save(detection: Detection): Result<Unit>
    fun observeAll(): Flow<List<Detection>>  // Flow para dados reativos
}

// ✅ Implementação na camada data
class DetectionRepositoryImpl @Inject constructor(
    private val remoteDataSource: DetectionRemoteDataSource,
    private val localDataSource: DetectionLocalDataSource,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : DetectionRepository {

    override suspend fun getAll(): Result<List<Detection>> =
        withContext(ioDispatcher) {
            runCatching {
                val remote = remoteDataSource.fetchDetections()
                localDataSource.saveAll(remote.map { it.toLocal() })
                remote.map { it.toDomain() }
            }
        }

    override fun observeAll(): Flow<List<Detection>> =
        localDataSource.observeAll()
            .map { entities -> entities.map { it.toDomain() } }
            .flowOn(ioDispatcher)
}
```

---

## Coroutines e Flow

```kotlin
// ✅ Dispatcher correto para cada operação
class ProcessVideoUseCase @Inject constructor(
    private val repository: DetectionRepository,
    @IoDispatcher private val ioDispatcher: CoroutineDispatcher,
) {
    suspend operator fun invoke(videoPath: String): Result<DetectionResult> =
        withContext(ioDispatcher) {
            runCatching {
                // I/O bound work no Dispatchers.IO
                processVideoInternal(videoPath)
            }
        }
}

// ✅ Flow com catch e retry
fun observeDetections(): Flow<List<Detection>> =
    repository.observeAll()
        .catch { e -> emit(emptyList()) }  // não deixe o flow quebrar
        .retry(3) { e -> e is NetworkException }
        .distinctUntilChanged()

// ❌ Proibido: GlobalScope (sem lifecycle management)
GlobalScope.launch { ... }  // NUNCA

// ❌ Proibido: runBlocking em código de produção Android
runBlocking { ... }  // NUNCA na UI thread
```

---

## Room (Local Database)

```kotlin
// ✅ Entity (modelo de persistência — separado do domínio)
@Entity(tableName = "detections")
data class DetectionEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "label") val label: String,
    @ColumnInfo(name = "confidence") val confidence: Float,
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis(),
)

// ✅ DAO com Flow para observação reativa
@Dao
interface DetectionDao {
    @Query("SELECT * FROM detections ORDER BY created_at DESC")
    fun observeAll(): Flow<List<DetectionEntity>>

    @Query("SELECT * FROM detections WHERE id = :id")
    suspend fun findById(id: String): DetectionEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: DetectionEntity)

    @Transaction  // para operações atômicas
    suspend fun replaceAll(entities: List<DetectionEntity>) {
        clearAll()
        insertAll(entities)
    }
}
```

---

## Hilt — Injeção de Dependências

```kotlin
// ✅ Module
@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient =
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .build()

    @Provides
    @Singleton
    fun provideRetrofit(client: OkHttpClient): Retrofit =
        Retrofit.Builder()
            .baseUrl(BuildConfig.BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
}

// ✅ Qualificadores para Dispatchers
@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class IoDispatcher

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class MainDispatcher

@Module
@InstallIn(SingletonComponent::class)
object DispatcherModule {
    @Provides @IoDispatcher
    fun provideIoDispatcher(): CoroutineDispatcher = Dispatchers.IO

    @Provides @MainDispatcher
    fun provideMainDispatcher(): CoroutineDispatcher = Dispatchers.Main
}
```

---

## Testes

```kotlin
// ✅ ViewModel test com TestCoroutineScheduler
@OptIn(ExperimentalCoroutinesApi::class)
class DetectionViewModelTest {

    @get:Rule
    val mainDispatcherRule = MainDispatcherRule()

    private val fakeRepository = FakeDetectionRepository()
    private lateinit var viewModel: DetectionViewModel

    @Before
    fun setUp() {
        viewModel = DetectionViewModel(
            processVideoUseCase = ProcessVideoUseCase(fakeRepository),
        )
    }

    @Test
    fun `processVideo updates uiState with detections on success`() = runTest {
        // Arrange
        fakeRepository.detectionToReturn = listOf(Detection(id = "1", label = "knife"))

        // Act
        viewModel.processVideo("/path/to/video.mp4")
        advanceUntilIdle()

        // Assert
        val state = viewModel.uiState.value
        assertThat(state.isLoading).isFalse()
        assertThat(state.detections).hasSize(1)
        assertThat(state.error).isNull()
    }
}
```

---

## Checklist Android/Kotlin

- [ ] `minSdk` verificado antes de usar APIs novas
- [ ] ViewModel com `viewModelScope`, nunca `GlobalScope`
- [ ] Repository retorna `Result<T>` ou `Flow<T>`
- [ ] Dispatchers injetados (não hardcoded) para testabilidade
- [ ] Composables stateless — UiState gerenciado no ViewModel
- [ ] Room entities separadas dos domain models
- [ ] Hilt com `@Singleton` apenas onde necessário
- [ ] `collectAsStateWithLifecycle()` em vez de `collectAsState()` na UI
- [ ] `key` nas LazyLists para evitar recomposição desnecessária
- [ ] Testes de ViewModel com `runTest` e `advanceUntilIdle()`
