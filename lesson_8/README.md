# 🎨 Просунуті технології GAN для генерації зображень


## Опис проєкту 📄

Цей проєкт реалізує: 
- **DC GAN** для генерації зображень автомобілів з набору **Stanford Cars**.
- **DC GAN** для генерації зображень з набору CIFAR10
- **Wasserstein GAN** для генерації зображень з набору **Stanford Cars**.

  Крім того були здійснені спроби реалізувати:
  - Feedback-aware GAN, коли результат дискріминатора не дискретна чи постійна величина, а вектор фідбеку, який слугує розширенням латентного вектора. Не вдалось, але підхід був не правильний, функція втрат дискримінатора-критика не продумана.
  - 3D-aware GAN, модель яка враховує ракурс при генерації зображення. Ще вчиться.


### 🔍 **Основна мета**
- Генерувати фотореалістичні зображення автомобілів на основі латентного шуму.
- Використання генеративних моделей для навчання на CIFAR10 та на більш складних подібних наборах даних.

---

## 🚀 **Особливості**
1. **Генератор та дискримінатор** створені за допомогою **PyTorch**.
2. **DC GAN** більш просунута модель, яка використовує зворотню конволюцію для генерації зображень. Для більш складних наборів даних, де звичайні повнозвʼязні шари не справляються.
3. **Набір Stanford Cars** вже досить важкий. Через це навчання на цьому наборі відбувається поетапно по 100 епох. Це призводить до того, що тестові зображення кожні 100 епох відрізняються
4. Візуалізація результатів для Stanford Cars на **кожній епосі**. Для CIFAR10 - кожні 10 епох.

---

## ⚙️ **Спостереження**
- Стандартний DCGAN входить в комфортну зону, генеруючи авто одночасно з великої кількості ракурсів, це для нього достатній компроміс. Теоретично, ще кілька десятків тисяч епох могли б це виправити, але це занадто.
- WS-GAN при цьому генерує більш згладжені картинки, втім також часто страждає на цю саму хворобу.

## **Візуалізація навчання**

![GIF Demo](slideshow.gif)

## **Роздуми, що можна зробити краще**
- при навчанні дискримінатора, можливо також є сенс аугментувати навчальний датасет, втім це може призвести до того, що він буде обганяти генератор і в результаті задушить його.
---

Щоб погратись з моделлю, потрібно викликати gan_call.py, та ввести цифру від 0 до 9.
