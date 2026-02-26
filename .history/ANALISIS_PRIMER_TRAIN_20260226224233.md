# Análisis del primer entrenamiento (Basic CNN, sin transfer learning)

## Tu problema
Convertir **imágenes a color 64×64×3** en un **número entre 0 y 1** que representa el **ángulo de una flecha** en la imagen (0 = una vuelta completa, 0.25 = 90°, etc.).

En el código usas **sin/cos** del ángulo (salida de 2 valores) en lugar del ángulo directo, lo cual es una buena elección porque:
- El ángulo es circular (0 y 1 son el mismo punto).
- Predecir (sin, cos) con MSE evita discontinuidades y entrena mejor.

---

## Fallos encontrados y correcciones aplicadas

### 1. **Device (GPU) en `preprocess` y en el Dataset**
**Qué pasaba:**  
`preprocess()` devolvía `img.to(device)` y el `Dataset.__getitem__` también hacía `.to(device)`. Con `DataLoader(..., num_workers=8)` los workers son procesos separados; en Windows, CUDA y multiprocessing suelen dar problemas y los tensores en GPU no se serializan bien entre procesos.

**Corrección:**  
- `preprocess()` ahora devuelve el tensor en **CPU**.
- `ArrowsDS.__getitem__` devuelve tensores en **CPU**.
- El movimiento a GPU se hace solo en el bucle de entrenamiento (`batch_img.to(device)`), que corre en el proceso principal.

Así el Dataset y el DataLoader trabajan solo en CPU y el entrenamiento sigue usando la GPU correctamente.

---

### 2. **Weight decay demasiado alto (0.05)**
**Qué pasaba:**  
Un `weight_decay=0.05` penaliza mucho los pesos en cada paso. Con pocos datos y una red pequeña puede impedir que la red aprenda bien y que el loss baje de forma estable.

**Corrección:**  
Se ha cambiado a `weight_decay=0.01`. Si aún regulariza demasiado, puedes probar `1e-3` o `1e-4`.

---

### 3. **Typo: `print_interbal`**
**Qué pasaba:**  
El nombre de la variable era `print_interbal` en lugar de `print_interval`. No rompía la ejecución, pero hacía el código menos claro y podía dar lugar a errores al reutilizar la variable.

**Corrección:**  
Se ha renombrado a `print_interval` y se usa en todos los sitios (condición del `if` y en el `print` del loss).

---

### 4. **`num_workers=8` en Windows**
**Qué pasaba:**  
Con `num_workers > 0`, el DataLoader usa procesos hijos. En Windows eso puede provocar errores o bloqueos con CUDA (y antes con tensores en GPU en el Dataset).

**Corrección:**  
Se ha puesto `num_workers=0` para que la carga de datos sea en el mismo proceso que el entrenamiento. En Linux/Mac puedes probar de nuevo con `num_workers=4` o `8` si quieres más velocidad.

---

## Comprobaciones que ya estaban bien

- **CSV sin cabecera:** El archivo empieza por datos; usar `names=["path", "angle"]` es correcto.
- **Convención de ángulo:** `angle ∈ [0, 1]` y `sin/cos` con `angle * 2 * np.pi` es coherente.
- **Pérdida:** MSE entre (sin, cos) predichos y reales es adecuada para regresión circular.
- **test_error:** Uso de `arctan2(sin, cos)` y diferencia angular en grados (incluyendo el mínimo entre `diff` y `360 - diff`) es correcto.
- **Modelo:** Basic_cnn con entrada 3×64×64 y salida 2 (sin, cos) es coherente con el problema.

---

## Si sigues sin obtener buenos resultados

1. **Reejecutar desde cero**  
   Ejecuta todas las celdas en orden (sobre todo: carga de CSV → train_test_split → preprocess en train/test → creación de `ArrowsDS` y DataLoaders → modelo → optimizer → `train(...)`).

2. **Curva de aprendizaje**  
   Mira si el **loss de entrenamiento** baja.  
   - Si no baja: learning rate bajo, red pequeña o weight decay aún alto (prueba `weight_decay=1e-3`).  
   - Si baja pero el error en test no mejora: overfitting; puedes subir un poco el weight decay o añadir más datos (p. ej. la aumentación que tienes comentada).

3. **Normalizar la salida (opcional)**  
   El modelo devuelve dos números que no están en la circunferencia unidad. Para la **métrica** en grados puedes normalizar:  
   `pred_norm = pred / np.linalg.norm(pred, axis=1, keepdims=True)`  
   y luego usar `pred_norm` en `test_error` y en el cálculo del ángulo. El entrenamiento puede seguir con MSE sobre la salida sin normalizar.

4. **Más épocas**  
   Con 100 épocas y batch_size=16 deberías ver algo de aprendizaje. Si el loss se estabiliza pronto, puedes subir a 150–200 épocas y vigilar el error en test.

Con estos cambios, el primer entrenamiento (Basic CNN, sin transfer learning) debería ser estable y mostrar mejoras en loss y en error angular en test.
