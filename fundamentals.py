import re
from datetime import datetime
from io import StringIO

import pandas as pd

from db import get_connection


# ============================================================
# НАСТРОЙКИ
# ============================================================

TICKERS = [
    "SBER",
    "GAZP",
    "LKOH",
    "GMKN",
    "YDEX",
    "ROSN",
    "NVTK",
    "TATN",
    "MGNT",
    "MTSS",
    "MOEX",
    "VTBR",
    "AFLT",
    "ALRS",
    "CHMF",
    "NLMK",
    "PLZL",
    "PHOR",
    "SNGS",
    "SNGSP",
    "TCSG",
    "OZON",
    "RUAL",
    "PIKK",
    "IRAO",
]

# У некоторых бумаг slug Finam отличается от тикера.
PAGE_SLUGS = {
    "TCSG": "t",
}

START_YEAR = 2021
END_YEAR = 2025

REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    )
}


# ============================================================
# URL
# ============================================================

def get_page_url(ticker):
    slug = PAGE_SLUGS.get(
        ticker,
        ticker.lower(),
    )

    return (
        f"https://www.finam.ru/quote/moex/"
        f"{slug}/financial/"
    )


# ============================================================
# НОРМАЛИЗАЦИЯ ТЕКСТА
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\xa0", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# ФИНАНСОВЫЕ ЧИСЛА
# ============================================================

def parse_financial_number(value):
    """
    Finam указывает финансовые показатели в миллионах.

    Примеры:

        287 800
            -> 287800 млн руб.
            -> 287800 * 1e6

        1 246 тыс.
            -> 1246 тыс. млн руб.
            -> 1246 * 1e9

    Возвращается значение в рублях.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = normalize_text(value)

    if not text:
        return None

    if text.lower() in {
        "-",
        "—",
        "–",
        "нет",
        "н/д",
    }:
        return None

    text_lower = text.lower()

    multiplier = 1e6

    if "трлн" in text_lower:
        multiplier = 1e12

    elif "млрд" in text_lower:
        multiplier = 1e9

    elif "тыс" in text_lower:
        multiplier = 1e9

    text = re.sub(
        r"(?i)(трлн\.?|млрд\.?|тыс\.?)",
        "",
        text,
    )

    text = text.replace(
        "%",
        "",
    )

    text = text.replace(
        "\xa0",
        " ",
    )

    text = text.replace(
        "\u202f",
        " ",
    )

    text = text.replace(
        " ",
        "",
    )

    negative = False

    if (
        text.startswith("(")
        and text.endswith(")")
    ):
        negative = True
        text = text[1:-1]

    text = text.replace(
        "−",
        "-",
    )

    text = text.replace(
        "–",
        "-",
    )

    text = text.replace(
        "—",
        "-",
    )

    if "," in text and "." not in text:
        text = text.replace(
            ",",
            ".",
        )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    try:
        number = float(
            match.group(0)
        )

        if negative:
            number = -number

        return number * multiplier

    except (ValueError, TypeError):
        return None


# ============================================================
# КОЭФФИЦИЕНТЫ
# ============================================================

def parse_ratio(value):
    """
    Парсинг P/B и других коэффициентов.

    Например:
        1,18 -> 1.18
        0,55 -> 0.55
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = normalize_text(value)

    if not text:
        return None

    if text.lower() in {
        "-",
        "—",
        "–",
        "нет",
        "н/д",
    }:
        return None

    text = text.replace(
        ",",
        ".",
    )

    text = text.replace(
        "%",
        "",
    )

    text = text.replace(
        " ",
        "",
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    try:
        return float(
            match.group(0)
        )

    except (ValueError, TypeError):
        return None


# ============================================================
# СТОЛБЦЫ-ГОДЫ
# ============================================================

def get_year_columns(df):
    result = {}

    for column in df.columns:

        text = normalize_text(
            column
        )

        match = re.search(
            r"\b(20\d{2})\b",
            text,
        )

        if not match:
            continue

        year = int(
            match.group(1)
        )

        if (
            START_YEAR
            <= year
            <= END_YEAR
        ):
            result[year] = column

    return result


# ============================================================
# ПОИСК СТРОКИ
# ============================================================

def find_row(
    df,
    keywords,
):

    if df is None or df.empty:
        return None

    first_column = df.columns[0]

    # 1. Точное совпадение
    for index, value in df[first_column].items():

        label = normalize_text(
            value
        ).lower()

        for keyword in keywords:

            keyword = keyword.lower()

            if label == keyword:
                return index

    # 2. Начинается с
    for index, value in df[first_column].items():

        label = normalize_text(
            value
        ).lower()

        for keyword in keywords:

            keyword = keyword.lower()

            if label.startswith(keyword):
                return index

    # 3. Содержит
    for index, value in df[first_column].items():

        label = normalize_text(
            value
        ).lower()

        for keyword in keywords:

            keyword = keyword.lower()

            if keyword in label:
                return index

    return None


# ============================================================
# ЗНАЧЕНИЯ СТРОКИ
# ============================================================

def get_row_values(
    df,
    row_index,
):

    if row_index is None:
        return {}

    year_columns = get_year_columns(
        df
    )

    result = {}

    for year, column in year_columns.items():

        result[year] = df.loc[
            row_index,
            column,
        ]

    return result


# ============================================================
# SELENIUM
# ============================================================

def create_driver():

    try:

        from selenium import webdriver
        from selenium.webdriver.edge.options import Options

        options = Options()

        options.add_argument(
            "--headless=new"
        )

        options.add_argument(
            "--disable-gpu"
        )

        options.add_argument(
            "--no-sandbox"
        )

        options.add_argument(
            "--disable-dev-shm-usage"
        )

        options.add_argument(
            "--disable-blink-features=AutomationControlled"
        )

        options.add_argument(
            "--window-size=1920,1080"
        )

        options.add_argument(
            f"--user-agent={HEADERS['User-Agent']}"
        )

        options.page_load_strategy = "eager"

        driver = webdriver.Edge(
            options=options
        )

        return driver

    except Exception as edge_error:

        print(
            f"Edge WebDriver не запустился: "
            f"{edge_error}"
        )

        print(
            "Пробуем Chrome..."
        )

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()

        options.add_argument(
            "--headless=new"
        )

        options.add_argument(
            "--disable-gpu"
        )

        options.add_argument(
            "--no-sandbox"
        )

        options.add_argument(
            "--disable-dev-shm-usage"
        )

        options.add_argument(
            "--disable-blink-features=AutomationControlled"
        )

        options.add_argument(
            "--window-size=1920,1080"
        )

        options.add_argument(
            f"--user-agent={HEADERS['User-Agent']}"
        )

        options.page_load_strategy = "eager"

        try:

            driver = webdriver.Chrome(
                options=options
            )

            return driver

        except Exception as chrome_error:

            raise RuntimeError(
                "Не удалось запустить Edge или Chrome через Selenium.\n"
                f"Edge: {edge_error}\n"
                f"Chrome: {chrome_error}"
            )


# ============================================================
# ОЖИДАНИЕ ОТРИСОВКИ
# ============================================================

def wait_for_financial_page(
    driver
):

    from selenium.webdriver.support.ui import WebDriverWait

    def page_ready(d):

        html = d.page_source.lower()

        return (
            "чистая прибыль" in html
            and "активы" in html
            and "p/b" in html
        )

    WebDriverWait(
        driver,
        25,
    ).until(
        page_ready
    )


# ============================================================
# ЗАГРУЗКА HTML
# ============================================================

def load_rendered_html(
    driver,
    ticker,
):

    url = get_page_url(
        ticker
    )

    driver.get(
        url
    )

    wait_for_financial_page(
        driver
    )

    return driver.page_source


# ============================================================
# ПОИСК ТАБЛИЦ
# ============================================================

def identify_tables(tables):

    financial_table = None
    multiplier_table = None

    for table in tables:

        if table is None:
            continue

        if table.empty:
            continue

        table_text = " ".join(
            normalize_text(value).lower()
            for value
            in table.astype(str)
            .values.flatten()
        )

        has_financial_fields = (
            "чистая прибыль" in table_text
            and "активы" in table_text
        )

        has_equity = (
            "капитал на акционеров"
            in table_text
            or
            "собственный капитал"
            in table_text
        )

        if (
            has_financial_fields
            and has_equity
        ):
            financial_table = table

        if "p/b" in table_text:
            multiplier_table = table

    return (
        financial_table,
        multiplier_table,
    )


# ============================================================
# PARSE FINAM
# ============================================================

def parse_finam(
    driver,
    ticker,
):

    url = get_page_url(
        ticker
    )

    html = load_rendered_html(
        driver,
        ticker,
    )

    # Важно:
    # decimal="," и thousands=" "
    # нужны для корректного разбора:
    #
    # 1,18 -> 1.18
    # 5 578 -> 5578
    #
    tables = pd.read_html(
        StringIO(html),
        flavor="lxml",
        decimal=",",
        thousands=" ",
    )

    if not tables:

        raise ValueError(
            "После выполнения JavaScript "
            "таблицы Finam не найдены"
        )

    (
        financial_table,
        multiplier_table,
    ) = identify_tables(
        tables
    )

    if financial_table is None:

        raise ValueError(
            "Не найдена таблица финансовых показателей"
        )

    if multiplier_table is None:

        raise ValueError(
            "Не найдена таблица мультипликаторов P/B"
        )

    # ========================================================
    # ЧИСТАЯ ПРИБЫЛЬ
    # ========================================================

    net_income_row = find_row(
        financial_table,
        [
            "чистая прибыль",
        ],
    )

    if net_income_row is None:

        raise ValueError(
            "Не найден показатель "
            "'Чистая прибыль'"
        )

    # ========================================================
    # АКТИВЫ
    # ========================================================

    assets_row = find_row(
        financial_table,
        [
            "активы",
        ],
    )

    if assets_row is None:

        raise ValueError(
            "Не найден показатель "
            "'Активы'"
        )

    # ========================================================
    # КАПИТАЛ
    # ========================================================

    equity_row = find_row(
        financial_table,
        [
            "капитал на акционеров",
            "собственный капитал",
        ],
    )

    if equity_row is None:

        raise ValueError(
            "Не найден показатель "
            "'Капитал на акционеров'"
        )

    # ========================================================
    # СЫРЫЕ ДАННЫЕ
    # ========================================================

    net_income_raw = get_row_values(
        financial_table,
        net_income_row,
    )

    assets_raw = get_row_values(
        financial_table,
        assets_row,
    )

    equity_raw = get_row_values(
        financial_table,
        equity_row,
    )

    # ========================================================
    # P/B
    # ========================================================

    pb_row = find_row(
        multiplier_table,
        [
            "p/b",
        ],
    )

    if pb_row is None:

        raise ValueError(
            "Не найден показатель P/B"
        )

    pb_raw = get_row_values(
        multiplier_table,
        pb_row,
    )

    # ========================================================
    # RECORDS
    # ========================================================

    records = []

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        net_income = parse_financial_number(
            net_income_raw.get(year)
        )

        assets = parse_financial_number(
            assets_raw.get(year)
        )

        equity = parse_financial_number(
            equity_raw.get(year)
        )

        pb = parse_ratio(
            pb_raw.get(year)
        )

        # ----------------------------------------------------
        # MARKET CAP
        # ----------------------------------------------------

        market_cap = None

        if (
            equity is not None
            and equity > 0
            and pb is not None
            and pb > 0
        ):

            market_cap = (
                equity * pb
            )

        # ----------------------------------------------------
        # BOOK TO MARKET
        # ----------------------------------------------------

        book_to_market = None

        if (
            equity is not None
            and equity > 0
            and market_cap is not None
            and market_cap > 0
        ):

            book_to_market = (
                equity / market_cap
            )

        # ----------------------------------------------------
        # PROFITABILITY
        # ----------------------------------------------------

        profitability = None

        if (
            net_income is not None
            and equity is not None
            and equity != 0
        ):

            profitability = (
                net_income / equity
            )

        records.append(
            {
                "ticker": ticker,

                "report_year": year,

                "period_end": datetime(
                    year,
                    12,
                    31,
                ).date(),

                "net_income": net_income,

                "assets": assets,

                "equity": equity,

                "pb": pb,

                "market_cap": market_cap,

                "book_to_market": book_to_market,

                "profitability": profitability,

                "investment": None,

                "source_url": url,
            }
        )

    df = pd.DataFrame(
        records
    )

    # ========================================================
    # INVESTMENT
    # ========================================================

    if not df.empty:

        df = (
            df.sort_values(
                [
                    "ticker",
                    "report_year",
                ]
            )
            .reset_index(drop=True)
        )

        previous_assets = (
            df.groupby("ticker")["assets"]
            .shift(1)
        )

        previous_year = (
            df.groupby("ticker")["report_year"]
            .shift(1)
        )

        normal_previous_year = (
            df["report_year"]
            == previous_year + 1
        )

        df["investment"] = (
            df["assets"]
            / previous_assets
            - 1
        )

        df.loc[
            ~normal_previous_year,
            "investment",
        ] = None

    return df


# ============================================================
# СОЗДАНИЕ RAW ТАБЛИЦЫ
# ============================================================

def create_table():

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                CREATE SCHEMA IF NOT EXISTS raw;
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS raw.finam_fundamentals (
                    id BIGSERIAL PRIMARY KEY,

                    ticker VARCHAR(20) NOT NULL,

                    report_year INTEGER NOT NULL,

                    period_end DATE NOT NULL,

                    net_income DOUBLE PRECISION,

                    assets DOUBLE PRECISION,

                    equity DOUBLE PRECISION,

                    pb DOUBLE PRECISION,

                    market_cap DOUBLE PRECISION,

                    book_to_market DOUBLE PRECISION,

                    profitability DOUBLE PRECISION,

                    investment DOUBLE PRECISION,

                    source_url TEXT,

                    loaded_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE (
                        ticker,
                        report_year
                    )
                );
                """
            )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def save_to_raw(df):

    if df.empty:
        return 0

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            sql = """
                INSERT INTO raw.finam_fundamentals (
                    ticker,
                    report_year,
                    period_end,
                    net_income,
                    assets,
                    equity,
                    pb,
                    market_cap,
                    book_to_market,
                    profitability,
                    investment,
                    source_url,
                    loaded_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                )
                ON CONFLICT (
                    ticker,
                    report_year
                )
                DO UPDATE SET
                    period_end =
                        EXCLUDED.period_end,

                    net_income =
                        EXCLUDED.net_income,

                    assets =
                        EXCLUDED.assets,

                    equity =
                        EXCLUDED.equity,

                    pb =
                        EXCLUDED.pb,

                    market_cap =
                        EXCLUDED.market_cap,

                    book_to_market =
                        EXCLUDED.book_to_market,

                    profitability =
                        EXCLUDED.profitability,

                    investment =
                        EXCLUDED.investment,

                    source_url =
                        EXCLUDED.source_url,

                    loaded_at =
                        CURRENT_TIMESTAMP;
            """

            for row in df.itertuples(
                index=False
            ):

                cur.execute(
                    sql,
                    (
                        row.ticker,
                        int(row.report_year),
                        row.period_end,

                        row.net_income,
                        row.assets,
                        row.equity,

                        row.pb,
                        row.market_cap,

                        row.book_to_market,
                        row.profitability,
                        row.investment,

                        row.source_url,
                    ),
                )

        conn.commit()

        return len(df)

    finally:

        conn.close()


# ============================================================
# ОЧИСТКА
# ============================================================

def clean_dataframe(df):

    if df.empty:
        return df

    numeric_columns = [
        "net_income",
        "assets",
        "equity",
        "pb",
        "market_cap",
        "book_to_market",
        "profitability",
        "investment",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df[
        df["net_income"].notna()
        | df["assets"].notna()
        | df["equity"].notna()
    ]

    return df.reset_index(
        drop=True
    )


# ============================================================
# ФОРМАТИРОВАНИЕ ДЛЯ КОНСОЛИ
# ============================================================

def format_number(
    value,
    decimals=2,
):
    """
    Обычное число без научной записи.

    Примеры:

        1246000000000
            -> 1 246 000 000 000

        0.223378
            -> 0,223378

        1.18
            -> 1,18
    """

    if value is None:
        return "—"

    try:

        if pd.isna(value):
            return "—"

    except (TypeError, ValueError):

        return "—"

    value = float(value)

    text = f"{value:,.{decimals}f}"

    # Английский формат:
    # 1,234,567.89
    #
    # превращаем в русский:
    # 1 234 567,89

    text = text.replace(
        ",",
        "@",
    )

    text = text.replace(
        ".",
        ",",
    )

    text = text.replace(
        "@",
        " ",
    )

    # Убираем ненужные нули
    if "," in text:

        text = text.rstrip(
            "0"
        ).rstrip(
            ","
        )

    return text


def format_money(value):
    """
    Финансовые значения в абсолютных рублях.

    Например:

        1.246e12

    превращается в:

        1 246 000 000 000
    """

    return format_number(
        value,
        decimals=0,
    )


def format_ratio(value):
    """
    Коэффициенты.

    Например:

        1.18 -> 1,18
        0.8475 -> 0,8475
    """

    return format_number(
        value,
        decimals=4,
    )


def format_percent(value):
    """
    Доля в проценты.

    Например:

        0.223378 -> 22,34%
    """

    if value is None:
        return "—"

    try:

        if pd.isna(value):
            return "—"

    except (TypeError, ValueError):

        return "—"

    return (
        format_number(
            float(value) * 100,
            decimals=2,
        )
        + "%"
    )


# ============================================================
# ПРОВЕРКА P/B
# ============================================================

def validate_pb(
    ticker,
    df,
):

    if df.empty:
        return

    pb_values = (
        df["pb"]
        .dropna()
    )

    if pb_values.empty:
        return

    # Если медиана > 30,
    # вероятно снова неправильно разобран
    # десятичный разделитель.
    if pb_values.median() > 30:

        print(
            f"ПРЕДУПРЕЖДЕНИЕ: {ticker}: "
            f"подозрительно большие значения P/B."
        )


# ============================================================
# ВЫВОД РЕЗУЛЬТАТА
# ============================================================

def print_result(
    ticker,
    df,
):

    print()

    if df.empty:

        print(
            f"{ticker}: данных нет"
        )

        return

    print(
        f"{ticker}: получено "
        f"{len(df)} строк "
        f"({df['report_year'].min()}–"
        f"{df['report_year'].max()})"
    )

    print()

    print(
        f"{'Год':<6}"
        f"{'Чистая прибыль':>24}"
        f"{'Активы':>24}"
        f"{'Капитал':>24}"
        f"{'P/B':>12}"
        f"{'Капитализация':>24}"
        f"{'B/M':>12}"
        f"{'Profit.':>12}"
        f"{'Invest.':>12}"
    )

    print(
        "-" * 150
    )

    for row in df.itertuples(
        index=False
    ):

        print(
            f"{row.report_year:<6}"
            f"{format_money(row.net_income):>24}"
            f"{format_money(row.assets):>24}"
            f"{format_money(row.equity):>24}"
            f"{format_ratio(row.pb):>12}"
            f"{format_money(row.market_cap):>24}"
            f"{format_ratio(row.book_to_market):>12}"
            f"{format_percent(row.profitability):>12}"
            f"{format_percent(row.investment):>12}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "ЗАГРУЗКА ФУНДАМЕНТАЛЬНЫХ ДАННЫХ FINAM"
    )

    print(
        "=" * 80
    )

    create_table()

    success_count = 0
    error_count = 0

    all_data = []

    driver = None

    try:

        print()

        print(
            "Запуск браузера..."
        )

        driver = create_driver()

        print(
            "Браузер запущен"
        )

        for ticker in TICKERS:

            print()

            print(
                "-" * 80
            )

            print(
                ticker
            )

            try:

                df = parse_finam(
                    driver,
                    ticker,
                )

                df = clean_dataframe(
                    df
                )

                if df.empty:

                    print(
                        f"{ticker}: данных нет"
                    )

                    error_count += 1

                    continue

                validate_pb(
                    ticker,
                    df,
                )

                saved = save_to_raw(
                    df
                )

                print(
                    f"{ticker}: OK | "
                    f"строк: {len(df)} | "
                    f"сохранено: {saved}"
                )

                print_result(
                    ticker,
                    df,
                )

                all_data.append(
                    df
                )

                success_count += 1

            except Exception as e:

                print(
                    f"{ticker}: ОШИБКА — {e}"
                )

                error_count += 1

    finally:

        if driver is not None:
            driver.quit()

    print()

    print(
        "=" * 80
    )

    print(
        f"Успешно: {success_count}"
    )

    print(
        f"Ошибок: {error_count}"
    )

    print(
        "=" * 80
    )

    if all_data:

        result = pd.concat(
            all_data,
            ignore_index=True,
        )

        print()

        print(
            "ИТОГО"
        )

        print(
            f"Компаний: "
            f"{result['ticker'].nunique()}"
        )

        print(
            f"Строк: "
            f"{len(result)}"
        )

        print(
            f"Годы: "
            f"{result['report_year'].min()}–"
            f"{result['report_year'].max()}"
        )

        print()

        valid_pb = result[
            "pb"
        ].dropna()

        if not valid_pb.empty:

            print(
                "Проверка P/B:"
            )

            print(
                f"min    = "
                f"{format_ratio(valid_pb.min())}"
            )

            print(
                f"max    = "
                f"{format_ratio(valid_pb.max())}"
            )

            print(
                f"median = "
                f"{format_ratio(valid_pb.median())}"
            )


if __name__ == "__main__":
    main()