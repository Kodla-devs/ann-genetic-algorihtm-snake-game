# Snake Game — GA + YSA (TR)

Bu repo, **Genetik Algoritma (GA)** ve **Yapay Sinir Ağı (YSA)** ile kendi kendine oynamayı öğrenen klasik **Snake** oyununu içerir.  
Görselleştirme **pygame** ile yapılır; yılan çevresini 21 öznitelikli bir vektörle algılar, ağın çıktısına göre **düz / sağ / sol** kararları verir ve her jenerasyonda evrimleşir.

> Ağ mimarisi varsayılan olarak **[21, 16, 3]** (giriş, gizli, çıkış) yapıdadır.

---

## 🔧 Kurulum (Python 3.10)

Aşağıdaki adımlar, kullanıcı tarafından paylaşılan venv bilgileri ve paket sürümlerine göre hazırlanmıştır.

### 1) Yeni venv oluştur

```bash
# snake_game klasörünün içindeyken
python3.10 -m venv venv
```

Bu komut, `snake_game/venv` içinde bir sanal ortam kurar.

### 2) Ortamı aktive et

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scriptsctivate
```

### 3) Paketleri sürümleriyle kur

Aşağıdaki sürümler, sisteminizdeki `site-packages` referans alınarak sabitlenmiştir:

```bash
pip install joblib==1.2.0
pip install llvmlite==0.39.1
pip install numba==0.56.4
pip install numpy==1.23.5
pip install pygame==2.1.2
```

> `pip==25.2` ve `setuptools` genellikle otomatik gelir. İsterseniz `setuptools`’u eşitleyebilirsiniz:

```bash
pip install setuptools==63.2.0
```

---

## 📂 Proje Yapısı

- `main.py` — Çalıştırma senaryoları (hazır ağla “izlet”, elle oyna, eğit).
- `game.py` — Oyun döngüsü, görünür/görünmez çalışma modları, klavye kontrolleri.
- `snake.py` — Yılanın durumu, karar verme, uygunluk (fitness) ve çizim.
- `map.py` — Harita, gıda yerleştirme, ışın (ray) tabanlı tarama (21 girişin üretimi).
- `neural_network.py` — YSA tanımı, ileri besleme, **kaydet/yükle**.
- `genetic_algorithm.py` — GA akışı; ebeveyn seçimi, çaprazlama, mutasyon, değerlendirme.
- `constants.py` — Sabitler (boyutlar, yön vektörleri, görsel dosya yolları vs.).
- `img/` — `wall.png`, `apple.png`, `snake.png` gibi görseller.

---

## ▶️ Hızlı Başlangıç (main.py)

`main.py` içinde üç tip kullanım hazır gelir. İhtiyacınıza göre **yorum satırlarını açıp/kapatarak** mod seçersiniz.

### 1) Hazır ağı **izlet** (test modu)

Önceden eğitilmiş bir ağ ağırlık/bias dosyalarınız varsa (örn. `gen_100_weights.npy`, `gen_100_biases.npy`) şu satırlar **AÇIK** kalsın:

```python
from game import*
from genetic_algorithm import *

net = NeuralNetwork()
game = Game()

# test
net.load(filename_weights='gen_100_weights.npy', filename_biases='gen_100_biases.npy')
game.start(display=True, neural_net=net)
```

- `display=True` → Oyunu pencerede izlersiniz (klavye kullanmadan, YSA oynatır).
- `net.load(...)` → Ağırlık/bias dosyalarını yükler.

### 2) **Elle oyna** (klavyeyle)

Aşağıdaki iki satırı **açın**; üstteki “test” kısmını **kapatın**:

```python
# play
game = Game()
game.start(playable=True, display=True, speed=10)
```

- Yön tuşları: **Sağ/sol ok** ile yön kırarsınız (UP şu an kullanılmıyor).
- **ESC** ile çıkış.
- `speed` değeri görsel modda adım hızını kontrol eder (varsayılan `20`).

### 3) **Eğit** (GA)

Aşağıdaki iki satırı **açın**; diğer modları **kapatın**:

```python
# train
gen = GeneticAlgorithm(population_size=1000, crossover_method='neuron', mutation_method='weight')
gen.start()
```

- Varsayılan mimari `genetic_algorithm.py` içinde **[21, 16, 3]**’tür.
- Her jenerasyonda en iyi bireyin ağı **`gen_{N}_weights.npy` / `gen_{N}_biases.npy`** olarak kaydedilir.
- Değerlendirme, performans için **görünmez modda** yapılır (pencere açılmaz).

> **İpucu:** Eğitim sırasında pencerede izlemek isterseniz değerlendirme çağrılarını `display=True` ile yapabilirsiniz; fakat bu **çok yavaşlatır**. Eğitim için görünmez mod önerilir.

---

## 🧪 GA Değerlendirme: Paralel çalıştırma ve çoklu deneme ortalaması

`genetic_algorithm.py` içindeki `evaluation(...)` fonksiyonu, her ağı **birden çok kez** koşturup skorların **ortalamasını** alır.  
Aşağıdaki yorum satırları **SEKANSİYEL** (tek çekirdek) denemelerdir:

```python
# for i in range(len(networks)):
#     results1.append(game.start(display=True, neural_net=networks[i]))
#     results2.append(game.start(display=True, neural_net=networks[i]))
#     results3.append(game.start(display=True, neural_net=networks[i]))
#     results4.append(game.start(display=True, neural_net=networks[i]))
```

Bunların yerine **joblib** ile **paralel** (tüm çekirdekler) koşturma aktif edilmiştir:

```python
results1 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
results2 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
results3 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
results4 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
for i in range(len(results1)):
    networks[i].score = int(np.mean([results1[i], results2[i], results3[i], results4[i]]))
```

- `n_jobs=num_cores` CPU çekirdek sayısını otomatik kullanır.
- Her ağ 4 kez denenir; ortalama skor, tek atış kaynaklı rastgeleliği azaltır.
- **Not:** Paralel çalışırken görünür pencere (**display=True**) açmayın; başsız (headless) modda kalın.

---

## 💾 Ağı kaydet / yükle ve `allow_pickle=True` notu

`neural_network.py` içindeki `load(...)` şu şekildedir:

```python
# self.weights = np.load(filename_weights)
# self.biases = np.load(filename_biases)
self.weights = np.load(filename_weights, allow_pickle=True)
self.biases  = np.load(filename_biases, allow_pickle=True)
```

Ağırlık ve biaslar **NumPy array’leri listesi** olarak saklandığı için `np.save` sonucu **pickle’lı nesne** biçimi içerir. Modern NumPy sürümlerinde güvenlik nedeniyle `allow_pickle=False` varsayılandır; bu yüzden **yüklerken** `allow_pickle=True` verilmelidir. Aksi halde `Object arrays cannot be loaded when allow_pickle=False` hatası alırsınız.

`save(...)` ise skora göre otomatik adlandırma veya özel isimle kayıt yapar:

- Otomatik: `saved_weights_{score}.npy`, `saved_biases_{score}.npy`
- Özel: `name + '_weights.npy'`, `name + '_biases.npy'`

GA, her jenerasyonun en iyisini **`gen_{N}_weights.npy` / `gen_{N}_biases.npy`** olarak kaydeder.

---

## 🕹️ Oyun Mekaniği ve Kontroller

- **Girdi (21)**: Yılan, 7 doğrultuda (geri hariç) **duvar / beden / yemek** durumlarını tarayarak 21 öznitelik üretir.
- **Çıkış (3)**: `0 → düz`, `1 → sağ`, `2 → sol` kararları.
- **Uygunluk (fitness)**: `len(body)^2 * age` — uzun yaşayıp büyümeyi teşvik eder.
- **Açlık sayacı**: `starve = 500` — amaçsız dolanmayı cezalandırır.
- **Klavye** (playable mod): Sağ/Sol ok ile yön değiştir, **ESC** ile çık.

> Görseller `img/` klasöründen yüklenir (`constants.py` içinde yollar tanımlıdır).

---

## ⚙️ GA Parametreleri (özet)

`GeneticAlgorithm(..., generation_number=100, crossover_rate=0.3, crossover_method='neuron', mutation_rate=0.7, mutation_method='weight')`

- **population_size**: Birey sayısı (örn. `1000`)
- **generation_number**: Jenerasyon sayısı
- **crossover_rate / mutation_rate**: Üretilecek çocuk ve mutasyon adedi oranları
- **crossover_method**: `'weight'` veya `'neuron'`
- **mutation_method**: `'weight'` (ağırlık tabanlı mutasyon)

---

## ❗️ Sık Karşılaşılan Sorunlar

- **`pygame` penceresi açılmıyor / hata**: Ekransız sunucuda `display=True` kullanmayın. Eğitim için görünmez mod yeterlidir.
- **`allow_pickle` hatası**: Yukarıdaki `load(...)` implementasyonundaki gibi `allow_pickle=True` kullanın.
- **Çok yavaş eğitim**: Paralel değerlendirme açık olmalı ve görünmez modda kalın. `population_size` ve `generation_number` ayarlarını düşürebilirsiniz.

---

## 📜 Lisans / Kullanım

Eğitim ve demo amaçlıdır; dilediğiniz gibi kullanabilir veya genişletebilirsiniz.
