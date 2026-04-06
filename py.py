#!/usr/bin/env python3

import argparse
import sys

import cv2
import numpy as np


# ============================================================
# TODO 1: Wczytanie pliku wideo
# ============================================================

def wczytaj_wideo(sciezka_wideo: str):
    cap = cv2.VideoCapture(sciezka_wideo)
    if not cap.isOpened():
        print(f"BŁĄD: Nie można otworzyć pliku wideo: {sciezka_wideo}", file=sys.stderr)
        sys.exit(1)
    return cap


# ============================================================
# TODO 2: Wyznaczanie dobrych punktów do śledzenia
# ============================================================

def wykryj_punkty(obraz_szary: np.ndarray):
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    punkty = cv2.goodFeaturesToTrack(obraz_szary, mask=None, **feature_params)
    return punkty


# ============================================================
# TODO 3: Śledzenie punktów metodą Lucas-Kanade
# ============================================================

def sledz_punkty(poprzedni_obraz, biezacy_obraz, poprzednie_punkty):
    if poprzednie_punkty is None or len(poprzednie_punkty) == 0:
        return None, None, None

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    nowe_punkty, status, blad = cv2.calcOpticalFlowPyrLK(
        poprzedni_obraz,
        biezacy_obraz,
        poprzednie_punkty,
        None,
        **lk_params,
    )
    return nowe_punkty, status, blad


# ============================================================
# TODO 4: Rysowanie aktualnych punktów
# ============================================================

def rysuj_punkty(obraz, punkty):
    if punkty is None:
        return obraz

    for pkt in punkty:
        x, y = pkt.ravel()
        cv2.circle(obraz, (int(x), int(y)), 5, (0, 255, 0), -1)

    return obraz


# ============================================================
# TODO 5: Rysowanie trajektorii
# ============================================================

def rysuj_trajektorie(maska, stare_punkty, nowe_punkty):
    if stare_punkty is None or nowe_punkty is None:
        return maska

    for stary, nowy in zip(stare_punkty, nowe_punkty):
        x_old, y_old = stary.ravel()
        x_new, y_new = nowy.ravel()
        cv2.line(
            maska,
            (int(x_old), int(y_old)),
            (int(x_new), int(y_new)),
            (0, 0, 255),
            2,
        )

    return maska


# ============================================================
# Główna pętla przetwarzania wideo
# ============================================================

def przetwarzaj_wideo(sciezka_wideo: str):
    cap = wczytaj_wideo(sciezka_wideo)

    poprawnie, pierwsza_klatka = cap.read()
    if not poprawnie:
        print("BŁĄD: Nie można odczytać pierwszej klatki filmu", file=sys.stderr)
        cap.release()
        return

    poprzedni_szary = cv2.cvtColor(pierwsza_klatka, cv2.COLOR_BGR2GRAY)
    poprzednie_punkty = wykryj_punkty(poprzedni_szary)
    maska_trajektorii = np.zeros_like(pierwsza_klatka)

    while True:
        poprawnie, klatka = cap.read()
        if not poprawnie:
            break

        biezacy_szary = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)

        if poprzednie_punkty is None or len(poprzednie_punkty) == 0:
            poprzednie_punkty = wykryj_punkty(poprzedni_szary)
            poprzedni_szary = biezacy_szary.copy()
            continue

        nowe_punkty, status, blad = sledz_punkty(
            poprzedni_szary,
            biezacy_szary,
            poprzednie_punkty,
        )

        if nowe_punkty is None or status is None:
            poprzednie_punkty = wykryj_punkty(biezacy_szary)
            poprzedni_szary = biezacy_szary.copy()
            continue

        dobre_nowe = nowe_punkty[status.flatten() == 1]
        dobre_stare = poprzednie_punkty[status.flatten() == 1]

        if len(dobre_nowe) == 0:
            poprzednie_punkty = wykryj_punkty(biezacy_szary)
            poprzedni_szary = biezacy_szary.copy()
            maska_trajektorii = np.zeros_like(klatka)
            continue

        dobre_nowe = dobre_nowe.reshape(-1, 1, 2)
        dobre_stare = dobre_stare.reshape(-1, 1, 2)

        maska_trajektorii = rysuj_trajektorie(
            maska_trajektorii,
            dobre_stare,
            dobre_nowe,
        )

        wynik = rysuj_punkty(klatka.copy(), dobre_nowe)
        wynik = cv2.add(wynik, maska_trajektorii)

        liczba_punktow = len(dobre_nowe)
        cv2.putText(
            wynik,
            f"Sledzone punkty: {liczba_punktow}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Optical Flow - śledzenie punktów", wynik)

        klawisz = cv2.waitKey(30) & 0xFF
        if klawisz in (ord("q"), 27):
            break

        poprzedni_szary = biezacy_szary.copy()
        poprzednie_punkty = dobre_nowe

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Funkcja główna
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LAB4 - Optical Flow, gotowe rozwiązanie")
    parser.add_argument("--video", required=True, help="Ścieżka do pliku wideo")
    args = parser.parse_args()

    przetwarzaj_wideo(args.video)


if __name__ == "__main__":
    main()
