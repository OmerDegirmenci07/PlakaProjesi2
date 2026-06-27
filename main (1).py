"""
Modul  : plaka_dedektoru.py
Kaynak : Arac-tanima  (plaka_tanima__1_.pt + orijinal betik)
Degisiklikler:
  - Sabit Windows dosya yollari kaldirildi; parametrik hale getirildi.
  - cv2.imshow / cv2.waitKey kaldirildi (headless Docker ortami).
  - JSON yazma kaldirildi; sonuc dict olarak return ediliyor.
  - segment bazli tespitler: video boyunca tum gecen plakalar toplanir,
    en yuksek skorlu plaka "ana plaka" olarak secilir.
  - tespitler listesi icin zaman damgasi (zaman_saniye) eklendi.
  - Tum etiket ve anahtarlar FTR dokumani kurallarina uygundur.
"""

import cv2
import re
import os
from collections import Counter

# EasyOCR ve YOLO sadece gerektiginde import edilecek
# (import hatasi kontrolu main.py katmaninda)
import easyocr
from ultralytics import YOLO

# -----------------------------------------------------------------------
# Yapilandirma sabitleri
# -----------------------------------------------------------------------
KARE_ATLAMA     = 3      # Her N karede bir isle
YOLO_ESIK       = 0.45   # YOLO guven esigi
OCR_ESIK        = 0.35   # EasyOCR guven esigi
BOS_KARE_ESIGI  = 20     # Kac bos kare = arac gitti

# Turkiye plaka regex (bosluksuz, buyuk harf)
PLAKA_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}\d{2,5}$"
)

# -----------------------------------------------------------------------
# Yardsimci fonksiyonlar
# -----------------------------------------------------------------------

def _plaka_isle(kirpilmis: "np.ndarray", reader: "easyocr.Reader") -> list:
    """Kirpilmis plaka bolgesine on isleme + OCR uygular."""
    if kirpilmis.size == 0:
        return []

    import cv2 as _cv2
    buyutulmus = _cv2.resize(kirpilmis, None, fx=2, fy=2,
                             interpolation=_cv2.INTER_CUBIC)
    gaussian   = _cv2.GaussianBlur(buyutulmus, (0, 0), 3)
    keskin     = _cv2.addWeighted(buyutulmus, 1.5, gaussian, -0.5, 0)
    gri        = _cv2.cvtColor(keskin, _cv2.COLOR_BGR2GRAY)
    yumusak    = _cv2.bilateralFilter(gri, 9, 15, 15)
    _, islenmis = _cv2.threshold(yumusak, 0, 255,
                                  _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
    return reader.readtext(
        islenmis,
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        detail=1,
        paragraph=False,
    )


def _temizle(metin: str) -> str:
    """OCR ciktisini Turkce karakter ve bosluk icermeyen buyuk harfe donusturur."""
    t = re.sub(r"\s+", "", metin).upper()
    for src, dst in [("Ç","C"),("Ğ","G"),("İ","I"),("Ö","O"),("Ş","S"),("Ü","U"),
                     ("Q","A"),("X","K"),("W","M")]:
        t = t.replace(src, dst)
    return t


def _segmentten_en_iyi(havuz: list) -> dict | None:
    """Plaka havuzundan en guvenilir plaka-skor ciftini secer."""
    if not havuz:
        return None

    frekanslar  = Counter(p["plaka"] for p in havuz)
    max_frekans = frekanslar.most_common(1)[0][1]
    dinamik_baj = max(2, max_frekans * 0.25)

    gecerli = []
    for plaka, frekans in frekanslar.most_common():
        if frekans >= dinamik_baj:
            maks_skor = max(p["skor"] for p in havuz if p["plaka"] == plaka)
            gecerli.append({"plaka": plaka, "frekans": frekans,
                             "skor": round(maks_skor, 2)})

    return gecerli[0] if gecerli else None


# -----------------------------------------------------------------------
# Ana sinif
# -----------------------------------------------------------------------

class PlakaDedektoru:
    """
    YOLOv8 + EasyOCR tabanli plaka tanima modulu.

    Parametreler
    ------------
    model_yolu : str
        YOLO plaka tespit modelinin agirlık dosyasi (.pt).
    gpu        : bool
        EasyOCR icin GPU kullanim tercihi.
    """

    def __init__(self, model_yolu: str, gpu: bool = True):
        print("[PlakaDedektoru] YOLO modeli yukleniyor...")
        self.model  = YOLO(model_yolu)
        print("[PlakaDedektoru] EasyOCR baslatiliyor (ilk seferinde indirme olabilir)...")
        try:
            self.reader = easyocr.Reader(["en"], gpu=gpu)
            print("[PlakaDedektoru] GPU ile EasyOCR aktif.")
        except Exception:
            self.reader = easyocr.Reader(["en"], gpu=False)
            print("[PlakaDedektoru] CPU ile EasyOCR aktif.")

    # ------------------------------------------------------------------
    def video_tara(self, video_path: str) -> dict:
        """
        Tum videoyu tarar; plaka tespitlerini dondurur.

        Donus
        -----
        {
            "ana_plaka"  : str,           # En yuksek skorlu plaka ("tespit_edilemedi")
            "ana_skor"   : float,         # 0.0 – 1.0
            "tespitler"  : [              # Her segmentteki plaka kayitlari
                {
                    "zaman_saniye"    : float,
                    "plaka"           : str,
                    "confidence_score": float
                }, ...
            ]
        }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {video_path}")

        fps            = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tum_araclar    = []   # Segment bazli en iyi plakalar
        segment_havuzu = []
        bos_kare_no    = 0
        segment_aktif  = False
        segment_no     = 0
        kare_sayaci    = 0

        # Segment baslangic zamanini takip et
        segment_bas_zaman = 0.0

        print("[PlakaDedektoru] Video isleniyor...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            kare_sayaci += 1

            if kare_sayaci % KARE_ATLAMA != 0:
                continue

            zaman_saniye = round(kare_sayaci / fps, 2)

            try:
                sonuclar = self.model(frame, conf=YOLO_ESIK, verbose=False)
            except Exception as e:
                print(f"  [UYARI] YOLO hatasi kare {kare_sayaci}: {e}")
                continue

            bu_karede_plaka = False

            for sonuc in sonuclar:
                for box in sonuc.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = max(0, x2), max(0, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    yolo_guven   = float(box.conf[0])
                    kirpilmis    = frame[y1:y2, x1:x2]
                    ocr_sonuclar = _plaka_isle(kirpilmis, self.reader)

                    for _, metin, ocr_skoru in ocr_sonuclar:
                        if ocr_skoru < OCR_ESIK:
                            continue
                        temiz = _temizle(metin)
                        if PLAKA_REGEX.match(temiz):
                            birlesik = round((ocr_skoru + yolo_guven) / 2, 2)
                            segment_havuzu.append({
                                "plaka"       : temiz,
                                "skor"        : birlesik,
                                "zaman_saniye": zaman_saniye,
                            })
                            bu_karede_plaka  = True
                            segment_aktif    = True
                            if not segment_havuzu or len(segment_havuzu) == 1:
                                segment_bas_zaman = zaman_saniye

            # Segment bitis kontrolu
            if not bu_karede_plaka and segment_aktif:
                bos_kare_no += 1
                if bos_kare_no >= BOS_KARE_ESIGI:
                    segment_no += 1
                    en_iyi = _segmentten_en_iyi(segment_havuzu)
                    if en_iyi:
                        en_iyi["zaman_saniye"] = segment_bas_zaman
                        tum_araclar.append(en_iyi)
                        print(f"  [Segment {segment_no}] Plaka: {en_iyi['plaka']}  "
                              f"Skor: {en_iyi['skor']}  t={segment_bas_zaman}s")
                    segment_havuzu  = []
                    bos_kare_no     = 0
                    segment_aktif   = False
            else:
                bos_kare_no = 0

        # Videoda kalan son segment
        if segment_havuzu:
            segment_no += 1
            en_iyi = _segmentten_en_iyi(segment_havuzu)
            if en_iyi:
                en_iyi["zaman_saniye"] = segment_bas_zaman
                tum_araclar.append(en_iyi)

        cap.release()
        print(f"[PlakaDedektoru] {segment_no} araç segmenti islendi.")

        # En yuksek skorlu plaka = ana plaka
        if tum_araclar:
            en_iyi_arac  = max(tum_araclar, key=lambda x: x["skor"])
            ana_plaka    = en_iyi_arac["plaka"]
            ana_skor     = en_iyi_arac["skor"]
        else:
            ana_plaka = "tespit_edilemedi"
            ana_skor  = 0.0

        return {
            "ana_plaka": ana_plaka,
            "ana_skor" : round(ana_skor, 2),
            "tespitler": [
                {
                    "zaman_saniye"    : a["zaman_saniye"],
                    "plaka"           : a["plaka"],
                    "confidence_score": a["skor"],
                }
                for a in tum_araclar
            ],
        }
