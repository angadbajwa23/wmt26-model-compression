#!/usr/bin/env bash
# Download WMT26 constrained training data for the 3 task language pairs:
#   ces-deu  (Czech -> German)
#   eng-zho  (English -> Simplified Chinese)
#   eng-ara  (English -> Egyptian Arabic)
#
# Usage:
#   bash data_prep/download_wmt26.sh [output_dir]
#   output_dir defaults to data/wmt26-train/

set -euo pipefail

OUT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/wmt26-train}"

echo "Output dir: $OUT_DIR"

# Install mtdata if needed
if ! python3 -c "import mtdata" 2>/dev/null; then
    echo "Installing mtdata..."
    pip install "mtdata[hf,xlsx]==0.5.1"
fi

mkdir -p "$OUT_DIR"

# ─── CES-DEU ────────────────────────────────────────────────────────────────
if [[ -d "$OUT_DIR/ces-deu" ]]; then
    echo "Skipping ces-deu (already downloaded)"
else
    echo "=== Downloading ces-deu ==="
    mtdata get -l ces-deu --no-fail --compress --no-merge -j 8 -o "$OUT_DIR/ces-deu" -tr \
        Statmt-news_commentary-18.1-ces-deu \
        Tilde-eesc-2017-ces-deu \
        Tilde-ema-2016-ces-deu \
        Tilde-ecb-2017-ces-deu \
        Tilde-rapid-2016-ces-deu \
        Facebook-wikimatrix-1-ces-deu \
        LinguaTools-wikititles-2014-ces-deu \
        OPUS-ccmatrix-v1-ces-deu \
        OPUS-dgt-v4-ces-deu \
        OPUS-ecb-v1-ces-deu \
        OPUS-ecdc-v20160316-ces-deu \
        OPUS-elitr_eca-v1-ces-deu \
        OPUS-elrc_417_swedish_work_environ-v1-ces-deu \
        OPUS-elrc_ec_europa-v1-ces-deu \
        OPUS-elrc_emea-v1-ces-deu \
        OPUS-elrc_euipo_2017-v1-ces-deu \
        OPUS-elrc_europarl_covid-v1-ces-deu \
        OPUS-elrc_eur_lex-v1-ces-deu \
        OPUS-elrc_eu_publications-v1-ces-deu \
        OPUS-elrc_information_portal-v1-ces-deu \
        OPUS-elrc_antibiotic-v1-ces-deu \
        OPUS-elrc_presscorner_covid-v1-ces-deu \
        OPUS-elrc_vaccination-v1-ces-deu \
        OPUS-elrc_wikipedia_health-v1-ces-deu \
        OPUS-emea-v3-ces-deu \
        OPUS-eubookshop-v2-ces-deu \
        OPUS-euconst-v1-ces-deu \
        OPUS-gnome-v1-ces-deu \
        OPUS-globalvoices-v2018q4-ces-deu \
        OPUS-jrc_acquis-v3.0-ces-deu \
        OPUS-kde4-v2-ces-deu \
        OPUS-multiccaligned-v1.1-ces-deu \
        OPUS-multiparacrawl-v9b-ces-deu \
        OPUS-nllb-v1-ces-deu \
        OPUS-neulab_tedtalks-v1-ces-deu \
        OPUS-opensubtitles-v2024-ces-deu \
        OPUS-php-v1-ces-deu \
        OPUS-qed-v2.0a-ces-deu \
        OPUS-ted2020-v1-ces-deu \
        OPUS-tanzil-v1-ces-deu \
        OPUS-tatoeba-v20230412-ces-deu \
        OPUS-tildemodel-v2018-ces-deu \
        OPUS-ubuntu-v14.10-ces-deu \
        OPUS-xlent-v1.2-ces-deu \
        OPUS-bible_uedin-v1-ces-deu \
        OPUS-wikimedia-v20230407-ces-deu \
        OPUS-tldr_pages-v20251124-ces-deu
fi

# ─── ENG-ZHO (Simplified Chinese) ───────────────────────────────────────────
if [[ -d "$OUT_DIR/eng-zho" ]]; then
    echo "Skipping eng-zho (already downloaded)"
else
    echo "=== Downloading eng-zho ==="
    mtdata get -l eng-zho --no-fail --compress --no-merge -j 8 -o "$OUT_DIR/eng-zho" -tr \
        Statmt-news_commentary-18.1-eng-zho \
        Statmt-wikititles-3-zho-eng \
        Statmt-ccaligned-1-eng-zho_CN \
        ParaCrawl-paracrawl-1_bonus-eng-zho \
        Facebook-wikimatrix-1-eng-zho \
        Neulab-tedtalks_train-1-eng-zho \
        ELRC-wikipedia_health-1-eng-zho \
        ELRC-hrw_dataset_v1-1-eng-zho \
        LinguaTools-wikititles-2014-eng-zho \
        OPUS-ccmatrix-v1-eng-zho \
        OPUS-elrc_3056_wikipedia_health-v1-eng-zho \
        OPUS-elrc_wikipedia_health-v1-eng-zho \
        OPUS-elrc_2922-v1-eng-zho \
        OPUS-eubookshop-v2-eng-zho \
        OPUS-multiun-v1-eng-zho \
        OPUS-nllb-v1-eng-zho \
        OPUS-php-v1-eng-zho \
        OPUS-qed-v2.0a-eng-zho \
        OPUS-spc-v1-eng-zho \
        OPUS-ted2020-v1-eng-zho \
        OPUS-tanzil-v1-eng-zho \
        OPUS-ubuntu-v14.10-eng-zho \
        OPUS-xlent-v1.2-eng-zho \
        OPUS-bible_uedin-v1-eng-zho \
        OPUS-infopankki-v1-eng-zho \
        OPUS-tico_19-v20201028-eng-zho \
        OPUS-wikimedia-v20230407-eng-zho \
        OPUS-alt-v20191206-eng-zho \
        OPUS-opensubtitles-v2016-eng-zho \
        OPUS-opus100_train-1-eng-zho \
        OPUS-paracrawl_bonus-v9-eng-zho \
        OPUS-ted2013-v1.1-eng-zho \
        OPUS-tldr_pages-v20251124-eng-zho \
        OPUS-unpc-v1.0-eng-zho
fi

# ─── ENG-ARA ────────────────────────────────────────────────────────────────
if [[ -d "$OUT_DIR/eng-ara" ]]; then
    echo "Skipping eng-ara (already downloaded)"
else
    echo "=== Downloading eng-ara ==="
    mtdata get -l eng-ara --no-fail --compress --no-merge -j 8 -o "$OUT_DIR/eng-ara" -tr \
        Statmt-news_commentary-18.1-ara-eng \
        Statmt-tedtalks-2_clean-eng-ara \
        Statmt-ccaligned-1-ara_AR-eng \
        Facebook-wikimatrix-1-ara-eng \
        LinguaTools-wikititles-2014-ara-eng \
        OPUS-ccmatrix-v1-ara-eng \
        OPUS-elrc_3083_wikipedia_health-v1-ara-eng \
        OPUS-elrc_wikipedia_health-v1-ara-eng \
        OPUS-elrc_2922-v1-ara-eng \
        OPUS-eubookshop-v2-ara-eng \
        OPUS-gnome-v1-ara-eng \
        OPUS-globalvoices-v2018q4-ara-eng \
        OPUS-hplt-v2-ara-eng \
        OPUS-kde4-v2-ara-eng \
        OPUS-multiccaligned-v1-ara-eng \
        OPUS-multihplt-v2-ara-eng \
        OPUS-multiun-v1-ara-eng \
        OPUS-nllb-v1-ara-eng \
        OPUS-opensubtitles-v2024-ara-eng \
        OPUS-qed-v2.0a-ara-eng \
        OPUS-ted2020-v1-ara-eng \
        OPUS-tatoeba-v20230412-ara-eng \
        OPUS-ubuntu-v14.10-ara-eng \
        OPUS-wikipedia-v1.0-ara-eng \
        OPUS-xlent-v1.2-ara-eng \
        OPUS-bible_uedin-v1-ara-eng \
        OPUS-infopankki-v1-ara-eng \
        OPUS-tico_19-v20201028-ara-eng \
        OPUS-wikimedia-v20230407-ara-eng \
        OPUS-neulab_tedtalks-v1-ara-eng \
        OPUS-opus100_train-1-ara-eng \
        OPUS-tanzil-v1-ara-eng \
        OPUS-ted2013-v1.1-ara-eng \
        OPUS-tldr_pages-v20251124-ara-eng \
        OPUS-unpc-v1.0-ara-eng
fi

echo ""
echo "Download complete. Convert to JSONL with:"
echo "  python3 data_prep/convert_mtdata_to_jsonl.py --input-dir $OUT_DIR --output-dir data/wmt26"
