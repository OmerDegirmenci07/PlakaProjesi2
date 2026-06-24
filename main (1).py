# ======================================================
# PLAKA TANIMA MODÜLÜ - 5G & Yapay Zeka ile Akıllı Yol Güvenliği Yarışması
# ======================================================

import cv2
import easyocr
import json
import os
import re
import sys
from collections import Counter

# ======================================================
# YAPILANDIRMA
# ======================================================
proje_klasoru = os.path.dirname(os.path.abspath(__file__))

yolo_model_yolu = os.path.join(proje_klasoru, "plaka_tanima.pt")
video_yolu      = os.path.join(proje_klasoru, "14674550_3840_2160_60fps.mp4")
cikti_json      = os.path.join(proje_klasoru, "results.json")

kare_atlama      = 3      # Her 3 karede bir işle
yolo_esik        = 0.45   # YOLO güven eşiği
ocr_esik         = 0.35   # OCR güven eşiği
bos_kare_esigi   = 20     # Kaç boş kare = araç gitti
plaka_regex      = r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}\d{2,5}$"

# ======================================================
# 1. MODELLERİ BAŞLAT
# ======================================================
print("Modeller yükleniyor, lütfen bekleyin...")

try:
    from ultralytics import YOLO
    yolo_model = YOLO(yolo_model_yolu)
except Exception as e:
    print(f"HATA: YOLO modeli yüklenemedi: {e}")
    sys.exit(1)

reader = easyocr.Reader(['en'], gpu=True)
print("Modeller hazır.\n")

# ======================================================
# 2. GÖRÜNTÜ ÖN İŞLEME
# ======================================================

def plaka_isle(kirpilmis_plaka):
    """Büyüt → Keskinleştir → Gri → Otsu → OCR"""
    if kirpilmis_plaka.size == 0:
        return []

    buyutulmus = cv2.resize(kirpilmis_plaka, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gaussian = cv2.GaussianBlur(buyutulmus, (0, 0), 3)
    keskin = cv2.addWeighted(buyutulmus, 1.5, gaussian, -0.5, 0)
    gri = cv2.cvtColor(keskin, cv2.COLOR_BGR2GRAY)
    yumusak = cv2.bilateralFilter(gri, 9, 15, 15)
    _, islenmis = cv2.threshold(yumusak, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return reader.readtext(
        islenmis,
        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        detail=1,
        paragraph=False
    )

def metni_temizle(metin):
    """OCR çıktısını normalize eder - PDF'deki kurallara uygun"""
    temiz = re.sub(r'\s+', '', metin).upper()
    temiz = temiz.replace('Ç','C').replace('Ğ','G').replace('İ','I') \
                 .replace('Ö','O').replace('Ş','S').replace('Ü','U')
    temiz = temiz.replace('Q','A').replace('X','K').replace('W','M')
    return temiz

def segmenti_isle(segment_havuzu, segment_no):
    """Segmentteki plaka havuzundan en güvenilir plakayı seçer"""
    if not segment_havuzu:
        return None

    frekanslar = Counter([p["plaka"] for p in segment_havuzu])
    sirali = frekanslar.most_common()
    max_frekans = sirali[0][1]
    dinamik_baraj = max(2, max_frekans * 0.25)

    gecerli = []
    for plaka, frekans in sirali:
        if frekans >= dinamik_baraj:
            skorlar = [p["skor"] for p in segment_havuzu if p["plaka"] == plaka]
            gecerli.append({
                "plaka": plaka,
                "frekans": frekans,
                "skor": round(max(skorlar), 2)
            })

    if gecerli:
        en_iyi = gecerli[0]
        print(f"\n{'='*45}")
        print(f"  ARAÇ #{segment_no} TESPİT EDİLDİ")
        print(f"  Plaka : {en_iyi['plaka']}")
        print(f"  Skor  : {en_iyi['skor']:.2f}  |  Okunma: {en_iyi['frekans']}")
        print(f"{'='*45}")
        return en_iyi

    return None

# ======================================================
# 3. VİDEO İŞLEME DÖNGÜSÜ
# ======================================================
print(f"Video açılıyor: {video_yolu}")

if not os.path.exists(video_yolu):
    print(f"HATA: Video bulunamadı → {video_yolu}")
    sys.exit(1)

cap = cv2.VideoCapture(video_yolu)
if not cap.isOpened():
    print("HATA: Video açılamadı.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video bilgisi: {toplam_kare} kare | {fps:.0f} FPS\n")

tum_araclar = []
segment_havuzu = []
bos_kare_sayisi = 0
segment_aktif = False
segment_no = 0
kare_sayaci = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kare_sayaci += 1

    if kare_sayaci % kare_atlama != 0:
        continue

    try:
        sonuclar = yolo_model(frame, conf=yolo_esik, verbose=False)
    except Exception as e:
        print(f"YOLO hatası kare {kare_sayaci}: {e}")
        continue

    bu_karede_plaka_var = False

    for sonuc in sonuclar:
        for box in sonuc.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_guven = float(box.conf[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(0, x2), max(0, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            kirpilmis_plaka = frame[y1:y2, x1:x2]
            ocr_sonuclar = plaka_isle(kirpilmis_plaka)

            for _, metin, ocr_skoru in ocr_sonuclar:
                if ocr_skoru < ocr_esik:
                    continue

                temiz_metin = metni_temizle(metin)

                if re.match(plaka_regex, temiz_metin):
                    birlesik_skor = (ocr_skoru + yolo_guven) / 2
                    segment_havuzu.append({"plaka": temiz_metin, "skor": birlesik_skor})
                    bu_karede_plaka_var = True
                    segment_aktif = True
                    print(f"  → Kare {kare_sayaci}: {temiz_metin} ({birlesik_skor:.2f})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not bu_karede_plaka_var and segment_aktif:
        bos_kare_sayisi += 1

        if bos_kare_sayisi >= bos_kare_esigi:
            segment_no += 1
            sonuc = segmenti_isle(segment_havuzu, segment_no)
            if sonuc:
                tum_araclar.append(sonuc)

            segment_havuzu = []
            bos_kare_sayisi = 0
            segment_aktif = False
    else:
        bos_kare_sayisi = 0

    gosterim = cv2.resize(frame, (1280, 720))
    cv2.imshow('Plaka Tanima', gosterim)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if segment_havuzu:
    segment_no += 1
    sonuc = segmenti_isle(segment_havuzu, segment_no)
    if sonuc:
        tum_araclar.append(sonuc)

cap.release()
cv2.destroyAllWindows()

# ======================================================
# 4. JSON ÇIKTISI
# ======================================================
print(f"\n{'='*45}")
print(f"  TOPLAM TESPİT EDİLEN ARAÇ: {len(tum_araclar)}")
print(f"{'='*45}")

if tum_araclar:
    for i, arac in enumerate(tum_araclar, 1):
        print(f"  {i}. Araç → {arac['plaka']} (Skor: {arac['skor']:.2f})")

    en_iyi_arac = max(tum_araclar, key=lambda x: x["skor"])
    nihai_plaka = en_iyi_arac["plaka"]
    nihai_skor = en_iyi_arac["skor"]
else:
    print("  Hiçbir araç tespit edilemedi.")
    nihai_plaka = ""
    nihai_skor = 0.0

# Format: "arac_bilgisi" ve "tespitler" anahtarları
cikis_verisi = {
    "video_id": os.path.basename(video_yolu),
    "arac_bilgisi": {
        "tip": "belirsiz",       # Diğer modüller dolduruyor
        "plaka": nihai_plaka,
        "renk": "belirsiz",      # Diğer modüller dolduruyor
        "confidence_score": round(float(nihai_skor), 4)
    },
    "tespitler": [
        {
            "zaman_saniye": 0.0,
            "kategori": "arac_plakasi",
            "etiket": arac["plaka"],
            "confidence_score": round(float(arac["skor"]), 4)
        }
        for arac in tum_araclar
    ]
}

with open(cikti_json, "w", encoding="utf-8") as f:
    json.dump(cikis_verisi, f, ensure_ascii=False, indent=2)

print(f"\n  JSON kaydedildi: {cikti_json}")
print(f"  Ana Araç Plakası: {nihai_plaka}")
print(f"  Güven Skoru: {nihai_skor:.4f}")
print(f"{'='*45}\n")
