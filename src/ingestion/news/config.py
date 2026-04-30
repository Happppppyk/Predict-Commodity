"""
Scoring constants for the news_v2 pipeline.

GKG V2Locations uses FIPS 10-4 ADM1 codes (e.g., "USIA" = Iowa).
GKG SourceCommonName is a bare domain (e.g., "reuters.com").
GKG DATE is int64 in YYYYMMDDHHmmss format.
"""

# Only extract articles published on or after 2020-01-01.
MIN_DATE_INT = 20200101000000

# ---------------------------------------------------------------------------
# Trusted sources used when choosing a canonical article within a duplicate cluster.
# Match against GKG SourceCommonName (lowercase domain).
# ---------------------------------------------------------------------------
TRUSTED_SOURCES = {
    # Official agencies
    "usda.gov",
    "fas.usda.gov",
    "nass.usda.gov",
    "ers.usda.gov",
    "fao.org",
    "eia.gov",
    "conab.gov.br",
    # Top financial wires (PPTX explicit)
    "reuters.com",
    "bloomberg.com",
}

# ---------------------------------------------------------------------------
# Tier 1 (S1C): 100% FP 도메인 67개 블랙리스트
# 출처: docs/audit/2026-04-27-filter-fp-audit.md §7.1
# 18일 backfill에서 통과 캐노니컬 ≥ 5건 + LLM 라벨 100% other인 도메인.
# ---------------------------------------------------------------------------
BLACKLIST_SOURCES = {
    # 연예/스포츠
    "iheart.com", "wrestlinginc.com", "wrestlezone.com", "f4wonline.com",
    "cagesideseats.com", "rotowire.com", "worldcasinodirectory.com",
    "igamingbusiness.com",
    # 일반 종합 / 대형 매체
    "cnn.com", "cnbc.com", "nbcnews.com", "abcnews.com", "fortune.com",
    "businessinsider.com", "dailymail.co.uk", "aol.co.uk", "upi.com",
    "theepochtimes.com", "irishtimes.com",
    # NBC 지역
    "nbcdfw.com", "nbcphiladelphia.com", "nbcwashington.com", "nbcnewyork.com",
    # 정치/오피니언
    "washingtonexaminer.com", "dailypolitical.com", "dailykos.com",
    "alternet.org", "rawstory.com", "breitbart.com", "redstate.com",
    "dailycaller.com", "natlawreview.com", "mondaq.com",
    # 안보
    "globalsecurity.org", "breakingdefense.com", "military.com",
    "australiandefence.com.au",
    # 환경 (대두 무관)
    "insideclimatenews.org",
    # 암호화폐
    "cointelegraph.com", "pymnts.com", "bitcoinmagazine.com",
    # 보도자료/스팸
    "themarketsdaily.com", "bignewsnetwork.com", "investegate.co.uk",
    "financialcontent.com", "jdsupra.com",
    # 지역 영문 매체 (대두 무관)
    "phnompenhpost.com", "tribune.com.pk", "theweek.in",
    "kaieteurnewsonline.com", "koreatimes.co.kr", "arabtimesonline.com",
    "timesofoman.com", "bssnews.net", "capitalfm.co.ke",
    "frontpageafricaonline.com", "businessghana.com", "allafrica.com",
    "progressive-charlestown.com", "clickondetroit.com", "wokv.com",
    "wsbradio.com", "whdh.com", "laist.com", "stripes.com",
    "fingerlakes1.com", "thegardenisland.com", "maritime-executive.com",
    "insurancejournal.com",
    # Tier 5-C: 풍자 매체
    "theonion.com",
    # Tier 5-D (2026-04-28): 호주 — 대두 0생산, 대두유 0수출 (시장 무관)
    "abc.net.au", "econews.com.au", "farmweekly.com.au",
    "queenslandcountrylife.com.au", "stockjournal.com.au", "stockandland.com.au",
    "theland.com.au", "northqueenslandregister.com.au",
    "portlincolntimes.com.au", "plainsproducer.com.au", "theroar.com.au",
    "sbs.com.au", "greatlakesadvocate.com.au",
    # Tier 5-D: 영국 / 아일랜드 / 스코틀랜드 — 대두유 무관
    "thescottishfarmer.co.uk", "farminguk.com", "plymouthherald.co.uk",
    "walesonline.co.uk", "metro.co.uk", "express.co.uk",
    "cravenherald.co.uk",
    # Tier 5-D 정정 (2026-04-28): paultan.org 제거.
    # 자동차 블로그지만 말레이시아 팜유 정책 (FELDA B100/B20 mandate) 실제 보도 있음.
    # Leapmotor B10 EV 같은 약자 충돌은 NEGATIVE_KEYWORDS_REGEX에서 별도 차단.
}

# ---------------------------------------------------------------------------
# Tier 1 (S1C): 음성 키워드 (false-friend) — 미래 데이터 대비 방어용
# 18일 backfill 측정에서는 0건 매치였으나 향후 confidence를 위해 유지.
# 출처: docs/audit/research_professional_practices.md §4.4 false-friend
# ---------------------------------------------------------------------------
NEGATIVE_KEYWORDS_REGEX = (
    # Tier 5-A 추가 (2026-04-28): recipe / Compulsory Purchase Order / DCE Ghana 등
    # Tier 5-B 추가: 건강 study / FDA allergy alert / CFTC 비-commodity / UCO Bank /
    #               REDII 패션 / The Onion satire 등 약자·context 충돌 차단
    r"\b("
    r"palm\s+(warbler|warblers|civet|civets|cockatoo|cockatoos|"
        r"sunday|beach|tree|trees|reading|reader|sized)|"
    r"palmoplantar|"
    r"ethanol\s+abuse|"
    r"crush\s+(syndrome|injury|fracture|fractures)|"
    # Tier 5-A: 식품 / 레시피 noise
    r"soy\s+(milk\s+latte|sauce\s+brand|candle|candles|wax|"
        r"sauce\s+(recipe|chicken|marinade|dressing|glaze)|"
        r"and\s+garlic\s+(glaze|chicken|sauce|marinade)|"
        r"glazed?\s+(chicken|beef|salmon|pork|tofu)|"
        r"marinade)|"
    r"chicken\s+sandwich|"
    # Tier 5-A: CPO / DCE 약자 충돌
    r"compulsory\s+purchase|CPO\s+(order|fears|process|notice)|"
    r"district\s+chief\s+executive|DCE\s+(orders?|probe|firing)|"
    # Tier 5-B: 건강 연구 / FDA allergy alert (recipe는 제외)
    r"soy.{1,60}(?:COPD|cancer|menopause|nutrition\s+study|health\s+(?:benefit|study)|"
        r"(?:improve|reduce|prevent)\s+\w+\s+symptoms?)|"
    r"(?:legume|legumes)\s+.{0,30}\bsoy\b|"
    r"allergy\s+alert|"
    r"(?:undeclared|recall(?:s|ed)?)\s+(?:milk|wheat|soy|allergen|tree\s+nuts?)|"
    # Tier 5-B + 5-C: CFTC 비-commodity context (.{0,150}로 relax — 긴 제목 대응)
    r"CFTC.{0,150}(precious\s+metal|silver\s+(?:depository|act)|gold\s+depository|"
        r"crypto(?:\s+watchdog|currency)?|AI\s+(?:staff|staffing|adopt)|bitcoin|"
        r"national\s+security\s+risks|geographic(?:al)?\s+concentration|"
        r"depositor(?:y|ies)|congress.{0,30}(?:budget|fees|money))|"
    # Tier 5-B: 약자 충돌 — UCO Bank (인도), REDII (패션)
    r"UCO\s+Bank|"
    r"REDII\s+(Works|FabLab)|"
    # Tier 5-C: B10/B20/B100 자동차 모델 충돌 (Leapmotor B10 EV 등)
    r"(?:Leapmotor|Geely|BYD|XPeng|Nio|Tesla|Hyundai|Kia)\s+B\d+|"
    r"B(?:10|20|100)\s+(?:EV|electric|car|model|sedan|SUV|rebate)|"
    r"commodore\s+(64|computer|computers)|"
    # Tier 5-F (2026-04-28 audit): docs/audit/filtered_v1_relevance_audit_2026-04-28.md §7.A
    # A1: HVO=Hawaii Volcano Observatory ≠ Hydrotreated Vegetable Oil (usgs.gov 23건 단일 클러스터)
    r"USGS\s+Volcano\s+Notice|"
    # A2: UCO=University of Central Oklahoma ≠ Used Cooking Oil
    r"UCO\s+(?:Research|Bronchos|honeybee|professor|department|university|grant)|"
    r"University\s+of\s+Central\s+Oklahoma|"
    # A4: biomethane/biogas/biofuel-HGV — 메탄·가스 carrier, vegetable oil 시장과 무관
    r"biomethane|"
    r"biogas\s+(?:plant|injection|grid|HGV|HGVs|truck|trucks|fleet|fleets)|"
    r"biofuel\s+HGVs?"
    r")\b"
)

# ---------------------------------------------------------------------------
# Tier 2 (S2C): V2Themes 화이트리스트 — 농업 anchor AND 결합용
# 출처: docs/audit/research_gdelt_fields.md §3.1 (실측 검증) + §3.3 (regex)
# Tier 1 themes: 단일 매치만으로도 신호 (real% ≥ 50, 18일 backfill 검증)
# Tier 2 themes: 보조 (다른 신호와 결합 시에만, real% 30-50)
# ---------------------------------------------------------------------------
V2THEMES_TIER1_REGEX = (
    r"\b("
    r"TAX_FOODSTAPLES_SOYBEANS?|"
    r"ENV_BIOFUEL|WB_532_BIOFUELS_ENERGY|WB_525_RENEWABLE_ENERGY|"
    r"NATURAL_DISASTER_(DROUGHTS?|FLOOD(ED|ING|S|WATERS)?|"
        r"HURRICANES?|MONSOON|HEATWAVES?|EXTREME_WEATHER|SEVERE_WEATHER|"
        r"TORRENTIAL_RAINS?|FLASH_FLOODS?|CYCLONES?)|"
    r"WB_(1774_CLIMATE_FORECASTING|571_CLIMATE_SCIENCE|"
        r"140_AGRICULTURAL_WATER_MANAGEMENT|151_IRRIGATION_AND_DRAINAGE|"
        r"526_RENEWABLE_ENERGY_POLICY)|"
    r"TAX_FNCACT_METEOROLOGIST|TAX_FOODSTAPLES_CORN"
    r")\b"
)

V2THEMES_TIER2_REGEX = (
    r"\b("
    r"AGRICULTURE|"
    r"TAX_FOODSTAPLES_(WHEAT|GRAINS?|RICE|MAIZE|LEGUMES|OLIVE_OIL)|"
    r"WB_(435_AGRICULTURE_AND_FOOD_SECURITY|1967_AGRICULTURAL_RISK_AND_SECURITY|"
        r"170_AGRICULTURAL_POLICIES|192_AGRICULTURAL_TRADE|"
        r"1949_CLIMATE_SMART_AGRICULTURE|174_CROP_PRODUCTION|177_ANIMAL_PRODUCTION|"
        r"175_FERTILIZERS|178_PEST_MANAGEMENT|199_FOOD_SECURITY|"
        r"2179_AGRIBUSINESS|1058_AGRIBUSINESS|2559_FOOD_TRADE|"
        r"1968_CROP_SELECTION_AND_PRODUCTION|1960_FARM_INPUTS|"
        r"1978_AGRICULTURE_AND_RELATED_SUBSIDIES|1959_LINKING_FARMERS_WITH_MARKETS|"
        r"2189_AGRICULTURE_INSURANCE|1972_POST_HARVEST_LOSSES|"
        r"567_CLIMATE_CHANGE|1773_CLIMATE_CHANGE_IMPACTS|"
        r"1777_FORESTS|1980_AGRO_FORESTRY)|"
    r"TAX_FNCACT_(FARMER|FARMERS|AGRONOMIST|DAIRY_FARMER|FARM_WORKERS?|FARMWORKER)|"
    r"FOOD_SECURITY|FOOD_STAPLE|RURAL|"
    r"CRISISLEX_O01_WEATHER|UNGP_AFFORDABLE_NUTRITIOUS_FOOD|"
    r"ECON_(OILPRICE|DIESELPRICE|TRADE_DISPUTE)|"
    r"ENV_DEFORESTATION|EPU_CATS_TRADE_POLICY"
    r")\b"
)

# ---------------------------------------------------------------------------
# Soybean filter — regex-based to avoid substring over-matching
# (e.g., naive ILIKE '%adm%' matches "administration", "admin", ...).
# ---------------------------------------------------------------------------

# Title: word-boundary regex for EN/PT/ES/UK/DE. KO matched separately via LIKE.
SOYBEAN_TITLE_REGEX = r"\b(soybean|soybeans|soy|soja|soya|sojabohne|sojabohnen)\b"
SOYBEAN_TITLE_KO = "대두"

# V2Organizations: canonical entity names. Matched with `(^|;)NAME,` boundary
# so ADM/LDC only hit when they appear as an actual entity, not a substring.
# GKG V2Organizations format: "Name1,offset1;Name2,offset2;..."
SOYBEAN_ORGS_EXACT = [
    "USDA", "CONAB", "ANEC", "Abiove",
    "Cargill", "Bunge", "ADM", "Archer Daniels Midland",
    "COFCO", "Louis Dreyfus", "LDC",
]


def soybean_org_regex() -> str:
    """Build a case-insensitive regex that matches a canonical org name
    as a complete entity in V2Organizations."""
    import re
    alts = "|".join(re.escape(n) for n in SOYBEAN_ORGS_EXACT)
    return rf"(^|;)({alts}),"


# ===========================================================================
# EXTENSION: BIOFUEL / CRUSH COMPLEX / SOUTH-AM CRUSHERS
# ===========================================================================
# Source: domain expert agent (a993a89e81eed9e6b, 2026-04-21)
# Covers RFS/RVO/SAF/HVO/renewable diesel, crush margin, Amaggi/Vicentin etc.

EXT_TITLE_REGEX_COMMON = (
    r"\b("
    r"RFS2?|RVO|SRE|LCFS|RINs?|"
    r"RED\s?I{1,3}|REPowerEU|ILUC|"
    r"RenovaBio|CBIOs?|"
    r"B1[0-5]|B20|B100|"
    r"HVO|SAF|UCO|"
    r"CBOT|MATBA-ROFEX|MATBA|ROFEX|DCE|B3|"
    r"WASDE|"
    r"ANEC|ABIOVE|CONAB|MAPA|ANP|"
    r"SAGyP|INDEC|"
    r"MOFCOM|Sinograin|"
    r"FAS|CFTC|USTR|IGC"
    r")\b"
)

EXT_TITLE_REGEX_EN = (
    # Tier 5-A 정정 (2026-04-28): `el niño|la niña` 단독 매치 제거.
    # Tier 5-B 회수: ENSO + 농업 키워드 결합 시 통과 (백업 손실 ~10건 회수 목표).
    r"\b("
    r"biodiesel|biofuels?|"
    r"renewable\s+diesel|renewable\s+fuel\s+standard|"
    r"sustainable\s+aviation\s+fuel|"
    r"small\s+refinery\s+exemptions?|"
    r"renewable\s+volume\s+obligations?|"
    r"low\s+carbon\s+fuel\s+standard|"
    r"soy(?:bean)?\s+(?:meal|oil|flour|protein|isolate)|"
    r"soymeal|soyoil|"
    r"crush\s+(?:margin|spread|rate|capacity)|"
    r"soy\s+crush|soybean\s+crush|"
    # A3 (audit 2026-04-28 §7.A): UCO 단독 매치 → 양방향 컨텍스트 결합으로 강화
    # binman 쓰레기 처리·자재화·학술 단발 보도 차단, biodiesel/feedstock 신호는 유지
    r"(?:used\s+cooking\s+oil.{0,40}(?:biodiesel|biofuel|renewable\s+diesel|HVO|RVO|RFS|RIN|FAME|feedstock|crush|tallow|SAF|sustainable\s+aviation\s+fuel|RenovaBio)"
    r"|(?:biodiesel|biofuel|renewable\s+diesel|HVO|RVO|RFS|RIN|FAME|feedstock|crush|tallow|SAF|sustainable\s+aviation\s+fuel|RenovaBio).{0,40}used\s+cooking\s+oil)|"
    r"(?:tallow.{0,20}(?:biodiesel|biofuel|renewable\s+diesel|HVO|UCO|crush|feedstock)"
    r"|(?:biodiesel|biofuel|renewable\s+diesel|HVO|UCO|crush|feedstock).{0,20}tallow)|"
    r"rapeseed\s+oil|canola\s+oil|"
    r"soybean\s+(?:rust|yield|acreage|planting|harvest|stocks?|exports?|imports?)|"
    r"sudden\s+death\s+syndrome|"
    r"prospective\s+plantings|grain\s+stocks|"
    r"wasde\s+report|crop\s+progress|"
    # Tier 5-B: ENSO + 농업 결합 (양방향, 복수형 허용)
    r"(?:el|la)\s+ni[ñn][oa].{0,80}"
        r"(?:crops?|farms?|farmers?|agri\w*|monsoons?|harvests?|soy\w*|palm\w*|"
        r"wheat|corn|maize|rice|grains?|food\s+(?:security|prices?|supply))|"
    r"(?:crops?|farms?|farmers?|agri\w*|monsoons?|harvests?|soy\w*|palm\w*|"
        r"wheat|corn|maize|rice|grains?|food\s+(?:security|prices?|supply))"
        r".{0,80}(?:el|la)\s+ni[ñn][oa]"
    r")\b"
)

EXT_TITLE_REGEX_PT = (
    # Tier 5-A 정정: `el niño|la niña` 단독 제거. `seca` 유지.
    # Tier 5-B 회수: ENSO + 농업 결합 추가.
    r"\b("
    r"biodiesel|biocombust[íi]veis?|"
    r"mistura\s+obrigat[óo]ria|"
    r"di[ée]sel\s+renov[áa]vel|"
    r"[óo]leo\s+de\s+soja|farelo\s+de\s+soja|"
    r"complexo\s+soja|complexo\s+da\s+soja|"
    r"esmagamento(?:\s+de\s+soja)?|"
    r"safra\s+(?:de\s+)?soja|"
    r"plantio\s+(?:de\s+)?soja|"
    r"colheita\s+(?:de\s+)?soja|"
    r"exporta[çc][õo]es?\s+de\s+soja|"
    r"seca|"
    # Tier 5-B: ENSO + 농업 결합
    r"(?:el|la)\s+ni[ñn][oa].{0,80}"
        r"(?:soja|safra|colheita|plantio|lavoura|agricultura|fazend|monç[ãa]o)|"
    r"(?:soja|safra|colheita|plantio|lavoura|agricultura|fazend|monç[ãa]o)"
        r".{0,80}(?:el|la)\s+ni[ñn][oa]"
    r")\b"
)

EXT_TITLE_REGEX_ES = (
    # Tier 5-A 정정: `el niño|la niña` 단독 제거. `sequía` 유지.
    # Tier 5-B 회수: ENSO + 농업 결합 추가.
    r"\b("
    r"biodi[ée]sel|biocombustibles?|"
    r"corte\s+de\s+biodi[ée]sel|"
    r"di[ée]sel\s+renovable|"
    r"aceite\s+de\s+soja|harina\s+de\s+soja|"
    r"complejo\s+sojero|complejo\s+de\s+la\s+soja|"
    r"molienda\s+de\s+soja|industria\s+aceitera|"
    r"cosecha\s+de\s+soja|siembra\s+de\s+soja|"
    r"campa[ñn]a\s+sojera|"
    r"exportaciones?\s+de\s+soja|"
    r"retenciones?(?:\s+a\s+la\s+soja)?|"
    r"derechos\s+de\s+exportaci[óo]n|"
    r"sequ[íi]a|"
    # Tier 5-B: ENSO + 농업 결합
    r"(?:el|la)\s+ni[ñn][oa].{0,80}"
        r"(?:soja|cosecha|siembra|cultivo|cosechad|agricult|campesino|maíz|trigo)|"
    r"(?:soja|cosecha|siembra|cultivo|cosechad|agricult|campesino|maíz|trigo)"
        r".{0,80}(?:el|la)\s+ni[ñn][oa]"
    r")\b"
)

# ---------------------------------------------------------------------------
# Macro signals — FX (BRL/ARS), crude oil × biofuel, sunflower oil substitute
# ---------------------------------------------------------------------------
# 환율과 원유는 단독 매치 시 일반 외환·에너지 보도 다량 유입 위험이 있어,
# soy/oilseed/grain/biodiesel 컨텍스트와 양방향 결합한다.
# 해바라기유는 oilseed substitute로 단독 허용(드물고 도메인 명확).
# Tier 2 V2Themes anchor (ECON_OILPRICE/DIESELPRICE/TRADE_DISPUTE 또는
# AGRICULTURE/FOOD_SECURITY)가 별도로 강제되므로 추가 FP 위험은 제한적.
EXT_TITLE_REGEX_MACRO = (
    r"\b("
    # 환율 페어 × 농업·바이오연료 컨텍스트 (양방향)
    r"(?:USD/BRL|BRL/USD|USD/ARS|ARS/USD).{0,80}"
        r"(?:soy\w*|soja|soybean|grain|oilseed|crush|export|safra|cosecha|harvest|biodiesel|biocombust)|"
    r"(?:soy\w*|soja|soybean|grain|oilseed|crush|export|safra|cosecha|harvest|biodiesel|biocombust)"
        r".{0,80}(?:USD/BRL|BRL/USD|USD/ARS|ARS/USD)|"
    # 통화명 × 농업·바이오연료 컨텍스트 (양방향)
    r"(?:Brazilian\s+real|argentine\s+peso|peso\s+argentin\w+).{0,80}"
        r"(?:soy\w*|soja|soybean|grain|oilseed|crush|export|safra|cosecha|harvest|biodiesel|biocombust)|"
    r"(?:soy\w*|soja|soybean|grain|oilseed|crush|export|safra|cosecha|harvest|biodiesel|biocombust)"
        r".{0,80}(?:Brazilian\s+real|argentine\s+peso|peso\s+argentin\w+)|"
    # 원유 × 바이오연료 breakeven/feedstock (양방향)
    r"(?:WTI|Brent|crude\s+oil|crude\s+prices?|oil\s+prices?).{0,40}"
        r"(?:biofuel|biodiesel|break-?even|feedstock|HVO|RVO|RFS|RIN|SAF|"
        r"renewable\s+diesel|sustainable\s+aviation\s+fuel)|"
    r"(?:biofuel|biodiesel|break-?even|feedstock|HVO|RVO|RFS|RIN|SAF|"
        r"renewable\s+diesel|sustainable\s+aviation\s+fuel).{0,40}"
        r"(?:WTI|Brent|crude\s+oil|crude\s+prices?|oil\s+prices?)|"
    # 해바라기유 (oilseed substitute) — oil은 단독 허용, seed는 컨텍스트 결합
    r"sunflower\s+oil|sunflowerseed\s+oil|"
    r"aceite\s+de\s+girasol|girasol\s+aceite|"
    r"[óo]leo\s+de\s+girassol|"
    r"sunflower\s+seeds?.{0,40}(?:market|exports?|imports?|crush|oilseed|commodity|prices?|futures|harvest)|"
    r"(?:market|exports?|imports?|crush|oilseed|commodity|prices?|futures|harvest).{0,40}sunflower\s+seeds?|"
    # 흑해 곡물 회랑 / 우크라이나 해바라기 공급 충격
    r"(?:black\s+sea|ukrain\w+).{0,40}"
        r"(?:sunflower|grain\s+corridor|grain\s+exports?|oilseed)"
    r")\b"
)


EXT_SOYBEAN_ORGS = [
    # US / international regulators & data agencies
    "Environmental Protection Agency",
    "Foreign Agricultural Service",
    "Commodity Futures Trading Commission",
    "Office of the United States Trade Representative",
    "International Grains Council",
    # Brazil
    "Ministério da Agricultura",
    "Agência Nacional do Petróleo",
    # Argentina
    "Secretaría de Agricultura",
    "Secretaría de Energía",
    "Instituto Nacional de Estadística y Censos",
    "Bolsa de Cereales",
    "Bolsa de Comercio de Rosario",
    "MATBA-ROFEX",
    # China
    "Ministry of Commerce",
    "Sinograin",
    "China National Cereals",
    # Global traders
    "Wilmar International", "Wilmar",
    "Olam", "Olam International", "Olam Agri",
    "Glencore Agriculture",
    "Viterra",
    "CHS",
    # Brazil crushers / traders
    "Amaggi", "Grupo Amaggi",
    "SLC Agrícola",
    "JBS",
    # Argentina crushers
    "Aceitera General Deheza",
    "Molinos Agro",
    "Molinos Río de la Plata",
    "Vicentin",
    "Bianchini",
    # Exchanges
    "Chicago Board of Trade",
    "Dalian Commodity Exchange",
    "CME Group",
    # Biofuel lobby
    "National Biodiesel Board",
    "Clean Fuels Alliance America",
    "Renewable Fuels Association",
    "Growth Energy",
]


def ext_soybean_org_regex() -> str:
    import re
    alts = "|".join(re.escape(n) for n in EXT_SOYBEAN_ORGS)
    return rf"(^|;)({alts}),"


# ===========================================================================
# EXTENSION: PALM OIL (soybean oil substitute — biodiesel + edible oil)
# ===========================================================================
# Source: domain expert agent (a7f27ede0532d1c2a, 2026-04-21)
# Indonesia B40/B50 정책, EU EUDR 2026-12 시행 반영

PALM_OIL_TITLE_REGEX_EN = (
    r"\b("
    r"palm\s+oils?|"
    r"crude\s+palm\s+oil|refined\s+palm\s+oil|"
    r"palm\s+kernel\s+oil|palm\s+kernels?|"
    r"palm\s+olein|palm\s+stearin|palm\s+fatty\s+acid|"
    r"RBD\s+palm|RBD\s+palm\s+olein|"
    r"palm\s+plantations?|oil\s+palms?|"
    r"palm\s+refiners?|palm\s+refining|"
    r"palm\s+exports?|palm\s+imports?|palm\s+shipments?|palm\s+stockpiles?|"
    r"palm\s+inventor(?:y|ies)|palm\s+production|palm\s+output|palm\s+yield|"
    r"palm\s+prices?|palm\s+futures|palm\s+complex|"
    r"palm\s+biodiesel|palm[-\s]based\s+biodiesel|palm\s+methyl\s+ester|"
    r"palm[-\s]soyoil\s+spread|soyoil[-\s]palm\s+spread|palm[-\s]soy\s+spread"
    r")\b"
)

PALM_OIL_TITLE_REGEX_PT = (
    r"\b("
    r"[óo]leo\s+de\s+palma|"
    r"[óo]leo\s+de\s+dend[êe]|"
    r"dend[êe]zeiro|dend[êe]icultura|"
    r"palma\s+de\s+[óo]leo|palmeira\s+de\s+[óo]leo|"
    r"[óo]leo\s+de\s+am[êe]ndoa\s+de\s+palma|"
    r"palmiste|ole[íi]na\s+de\s+palma|estearina\s+de\s+palma|"
    r"biodiesel\s+de\s+palma|biodiesel\s+de\s+dend[êe]"
    r")\b"
)

PALM_OIL_TITLE_REGEX_ES = (
    r"\b("
    r"aceite[s]?\s+de\s+palma|"
    r"aceite\s+crudo\s+de\s+palma|aceite\s+de\s+palmiste|"
    r"palma\s+aceitera|palma\s+africana|"
    r"ole[íi]na\s+de\s+palma|estearina\s+de\s+palma|"
    r"almendra\s+de\s+palma|"
    r"biodi[ée]sel\s+de\s+palma"
    r")\b"
)

PALM_OIL_COMMON = (
    r"\b("
    r"CPO|PKO|FCPO|RBDPO|PFAD|"
    r"B30|B35|B40|B50|"
    r"BMD|KLSE|"
    r"DMO|DPO|BPDPKS|MPOB|MPOC|MPOA|GAPKI|IPOA|RSPO|ISPO|MSPO|"
    r"EUDR"
    r")\b"
)

PALM_OIL_ORGS_EXACT = [
    # Malaysia producers
    "Sime Darby Plantation",
    "IOI Corporation",
    "IOI Corporation Berhad",
    "Kuala Lumpur Kepong",
    "Genting Plantations",
    "FGV Holdings",
    "Felda Global Ventures",
    "United Plantations",
    "Hap Seng Plantations",
    "TSH Resources",
    "Sarawak Oil Palms",
    # Indonesia producers (English name)
    "Golden Agri-Resources",
    "Astra Agro Lestari",
    "Indofood Agri Resources",
    "IndoAgri",
    "Musim Mas",
    "London Sumatra",
    "PP London Sumatra Indonesia",
    "Salim Ivomas Pratama",
    "Sampoerna Agro",
    "Triputra Agro Persada",
    "Bakrie Sumatera Plantations",
    "Dharma Satya Nusantara",
    "Austindo Nusantara Jaya",
    "Eagle High Plantations",
    # Brazil
    "Agropalma",
    # Global traders (palm + soy overlap)
    "Apical Group",
    "KLK Oleo",
    "Fuji Oil",
    "AAK",
    # Regulators / industry bodies
    "Malaysian Palm Oil Board",
    "Malaysian Palm Oil Council",
    "Malaysian Palm Oil Association",
    "Indonesian Palm Oil Association",
    "Roundtable on Sustainable Palm Oil",
    "Indonesian Sustainable Palm Oil",
    "Malaysian Sustainable Palm Oil",
    "Badan Pengelola Dana Perkebunan Kelapa Sawit",
    "Gabungan Pengusaha Kelapa Sawit Indonesia",
    # Exchanges
    "Bursa Malaysia Derivatives",
    "Bursa Malaysia",
]

# Acronyms for V2Organizations — CASE-SENSITIVE entity boundary.
# FGV excluded: conflicts with Fundação Getulio Vargas (brazilian think tank).
PALM_OIL_ORGS_ACRONYMS = [
    "MPOB", "MPOC", "MPOA", "GAPKI", "IPOA",
    "RSPO", "ISPO", "MSPO", "BPDPKS",
    "KLK", "GAR", "AAL",
]


def palm_oil_org_regex() -> str:
    import re
    alts = "|".join(re.escape(n) for n in PALM_OIL_ORGS_EXACT)
    return rf"(^|;)({alts}),"


def palm_oil_acronym_regex() -> str:
    import re
    alts = "|".join(re.escape(n) for n in PALM_OIL_ORGS_ACRONYMS)
    return rf"(^|;)({alts}),"


# ===========================================================================
# CATEGORY TAXONOMY — Codex scoring categories
# ===========================================================================
# Tier 5-E rollback (2026-04-28): snd_ar / weather_ar 제거.
# 이유: 18일 backfill에서 AR 관련 기사 0건 → schema에 카테고리만 두면 LLM 혼동.
# 데이터 누적 후 (예: 6개월) AR 기사 등장 시 재추가 검토.
CATEGORIES = [
    # --- 정책 (Policy) ---
    "policy_us_eia",        # US EIA biodiesel monthly report, 통계
    "policy_rvo",           # US RFS, RVO, SRE (EPA 규정)
    "policy_rebio",         # BR RenovaBio, CBIO, 혼합비 정책
    "policy_palm",          # ID/MY B35/B40, DMO, 수출제한·부과금
    # --- 작황 (Supply-n-Demand) ---
    "snd_us",               # USDA WASDE / NASS / FAS + 미국 농가 일반 보도
    "snd_br",               # Brazil CONAB, 작황, 수출
    "snd_palm",             # ID/MY 팜유 supply (MPOB monthly, GAPKI 등)
    # --- 기상 (Weather): 농업 영향 기상 ---
    "weather_us",           # Corn Belt / Mississippi Delta 가뭄·홍수·서리·허리케인
    "weather_br",           # Cerrado / Mato Grosso / RGS 기상
    "weather_global",       # AR Pampa, 인도 몬순, 인니 가뭄, 흑해 등
    # --- 시장 (Market) ---
    "market_general",       # 가격 코멘터리, crush margin, CBOT BO 시세
    "market_trade",         # 관세, USTR, MOFCOM, 미중 무역 합의
    "market_corporate",     # ABCD 어닝, M&A, 가공시설 신증설/폐쇄
    # --- 기타 ---
    "other",
]


# ---------------------------------------------------------------------------
# Language pipeline (scope narrowed to EN / PT / ES only)
# ---------------------------------------------------------------------------
LANGDETECT_CONFIDENCE_THRESHOLD = 0.65

# 파이프라인이 처리하는 언어 화이트리스트. 이 외 언어는 enriched_v1 단계에서 drop.
ALLOWED_LANGS = {"en", "pt", "es"}

# ---------------------------------------------------------------------------
# Strong domain override — 신뢰도 높은 TLD는 langdetect 결과를 덮어씀.
# langdetect가 짧은 CJK/세르비아어 title을 so/hr/ca로 오감지하는 문제 해결.
# ---------------------------------------------------------------------------
STRONG_DOMAIN_OVERRIDE = {
    # CJK
    ".com.cn": "zh", ".cn": "zh",
    "sina.com.cn": "zh", "sohu.com": "zh", "eastmoney.com": "zh",
    "163.com": "zh", "people.com.cn": "zh", "aweb.com.cn": "zh",
    "china.org.cn": "zh", "chinadaily.com.cn": "zh", "cfi.net.cn": "zh",
    # Slavic
    ".rs": "sr", ".ru": "ru", "chaspik.spb.ru": "ru",
    # Others where langdetect commonly fails
    ".vn": "vi",
}

# Weak fallback — langdetect 실패 시에만 참조.
DOMAIN_LANG_FALLBACK = {
    # pt
    ".com.br": "pt", ".pt": "pt",
    # es
    ".com.ar": "es", ".com.mx": "es", ".com.co": "es", ".com.pe": "es",
    ".com.es": "es", ".es": "es",
    # en
    "indiatimes.com": "en", ".co.uk": "en", ".com.au": "en",
    # zh / ru / sr (Strong override에서도 잡지만 보조)
    ".com.cn": "zh", ".cn": "zh",
    ".ru": "ru", ".rs": "sr",
    # others
    ".de": "de", ".fr": "fr", ".it": "it", ".nl": "nl",
    ".id": "id", ".tr": "tr",
    "dostor.org": "ar",
}


def _match_domain_suffix(source: str, mapping: dict[str, str]) -> str | None:
    if not source:
        return None
    s = source.lower()
    if s in mapping:
        return mapping[s]
    for suffix in sorted(mapping, key=len, reverse=True):
        if s.endswith(suffix):
            return mapping[suffix]
    return None


def strong_lang_for_source(source: str) -> str | None:
    """High-confidence TLD override — takes precedence over langdetect."""
    return _match_domain_suffix(source, STRONG_DOMAIN_OVERRIDE)


def fallback_lang_for_source(source: str) -> str | None:
    """Weak fallback — only consulted when langdetect fails/low-confidence."""
    return _match_domain_suffix(source, DOMAIN_LANG_FALLBACK)


# ---------------------------------------------------------------------------
# Central filter SQL builder — all scripts share the same WHERE clause so
# counts stay consistent with the final filtered parquet.
# ---------------------------------------------------------------------------
def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def soybean_filter_sql() -> str:
    """WHERE clause body for GKG rows (without leading 'WHERE').

    Tier 1 + Tier 2 (S2C) 적용 — docs/audit/2026-04-27-soybean-oil-filter-redesign.md.
    이전 (Tier 1): title 필수 + 음성 키워드 + 블랙리스트.
    추가 (Tier 2): V2Themes 농업 anchor 필수 (Tier 1 또는 Tier 2 화이트리스트 매치).

    측정 결과 (scripts/_audit_filter_ablation.py, 2026-04-27 새 universe):
      - S1C → S2C: precision 56.1% → 69.2% (+13pp, canonical 레벨)
      - real loss 7.7%
      - 'other' 47.6% 추가 차단 (canonical 336 → 176)

    Criteria:
      1. DATE >= 2020-01-01
      2. title is non-null / non-empty
      3. title 기반 10개 조건 중 ≥ 1개 매치
      4. V2Themes에 농업 anchor (Tier 1 OR Tier 2) ≥ 1개 매치 — Tier 2 핵심
      5. NOT 음성 키워드
      6. NOT 블랙리스트 도메인
    """
    # title 기반 (11개 조건 — Tier 5-G: 매크로 신호 추가)
    soy_title_regex = _sql_str(SOYBEAN_TITLE_REGEX)
    soy_title_ko = _sql_str("%" + SOYBEAN_TITLE_KO + "%")
    ext_en = _sql_str(EXT_TITLE_REGEX_EN)
    ext_pt = _sql_str(EXT_TITLE_REGEX_PT)
    ext_es = _sql_str(EXT_TITLE_REGEX_ES)
    ext_common = _sql_str(EXT_TITLE_REGEX_COMMON)
    ext_macro = _sql_str(EXT_TITLE_REGEX_MACRO)
    palm_en = _sql_str(PALM_OIL_TITLE_REGEX_EN)
    palm_pt = _sql_str(PALM_OIL_TITLE_REGEX_PT)
    palm_es = _sql_str(PALM_OIL_TITLE_REGEX_ES)
    palm_common = _sql_str(PALM_OIL_COMMON)

    # V2Themes 농업 anchor (Tier 2)
    theme_tier1 = _sql_str(V2THEMES_TIER1_REGEX)
    theme_tier2 = _sql_str(V2THEMES_TIER2_REGEX)

    # 음성 키워드 + 블랙리스트
    neg_kw = _sql_str(NEGATIVE_KEYWORDS_REGEX)
    blacklist_sql = ", ".join(_sql_str(d) for d in sorted(BLACKLIST_SOURCES))

    return (
        f"DATE >= {MIN_DATE_INT} "
        f"AND title IS NOT NULL AND TRIM(title) <> '' "
        # title 기반 11개 OR — org-only 매치 차단
        f"AND ("
        f"regexp_matches(title, {soy_title_regex}, 'i') "
        f"OR title LIKE {soy_title_ko} "
        f"OR regexp_matches(title, {ext_en}, 'i') "
        f"OR regexp_matches(title, {ext_pt}, 'i') "
        f"OR regexp_matches(title, {ext_es}, 'i') "
        f"OR regexp_matches(title, {ext_common}) "  # case-sensitive
        f"OR regexp_matches(title, {ext_macro}, 'i') "  # FX/crude/sunflower
        f"OR regexp_matches(title, {palm_en}, 'i') "
        f"OR regexp_matches(title, {palm_pt}, 'i') "
        f"OR regexp_matches(title, {palm_es}, 'i') "
        f"OR regexp_matches(title, {palm_common}) "  # case-sensitive
        f") "
        # V2Themes 농업 anchor 필수 (Tier 2) — case-sensitive (theme code는 UPPERCASE)
        f"AND ("
        f"regexp_matches(COALESCE(V2Themes, ''), {theme_tier1}) "
        f"OR regexp_matches(COALESCE(V2Themes, ''), {theme_tier2})"
        f") "
        # 음성 키워드 차단
        f"AND NOT regexp_matches(title, {neg_kw}, 'i') "
        # 블랙리스트 도메인 차단
        f"AND lower(SourceCommonName) NOT IN ({blacklist_sql})"
    )


# URL 정규화 DuckDB 식 (strip tracking params, fragment, trailing slash, lowercase).
NORMALIZED_URL_SQL = r"""
    rtrim(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(DocumentIdentifier),
                        '#.*$', ''
                    ),
                    '[?&](utm_[^&]*|fbclid=[^&]*|gclid=[^&]*|mc_cid=[^&]*|mc_eid=[^&]*)', '', 'g'
                ),
                '[?&]+$', ''
            ),
            '(\?)&', '\1', 'g'
        ),
        '/'
    )
"""


# ---------------------------------------------------------------------------
# GDELT API ingest (00_ingest.py)
# ---------------------------------------------------------------------------
GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
GDELT_LASTUPDATE_URL = f"{GDELT_BASE_URL}/lastupdate.txt"

# Hive-partitioned raw GKG parquet root: {GKG_RAW_DIR}/dt=YYYY-MM-DD/HHMMSS.parquet
import os as _os  # noqa: E402
# 이 파일 위치: <PROJECT_ROOT>/src/ingestion/news/config.py
# → PROJECT_ROOT = ../../../ 세 단계 위
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
GKG_RAW_DIR = _os.path.join(_PROJECT_ROOT, "data", "news", "gkg_raw")
