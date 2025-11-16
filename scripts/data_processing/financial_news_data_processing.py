from pathlib import Path
import pandas as pd
from typing import Any
from bs4 import BeautifulSoup
import html


def load_all_data(file_prefix: str, folder: str,  file_extension: str = "csv") -> pd.DataFrame:
    directory = Path(folder)
    csv_files = sorted(directory.glob(f"{file_prefix}*.{file_extension}"))
    if not csv_files:
        raise FileNotFoundError(f"No files matching '{file_prefix}*.{file_extension}' found in {directory.resolve()}")
    dfs = []
    for fp in csv_files:
        match file_extension:
            case "csv":
                df = pd.read_csv(fp)
            case "parquet":
                df = pd.read_parquet(fp)
            case _:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True, sort=False)
    try:
        combined_df['text'] = combined_df['text'].apply(extract_content_from_dict)
    except KeyError:
        pass
    return combined_df.dropna().drop_duplicates()


def extract_content_from_dict(row: Any) -> Any:
    if isinstance(row, dict):
        return row.get('content', '')
    else:
        return row

def clean_html(text: str) -> str:
    if not isinstance(text, str):
        return text
    bs = BeautifulSoup(text, "html.parser")
    for tag in bs(["script", "style"]):
        tag.decompose()
    text = bs.get_text(separator=" ", strip=True)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text


def load_news_data(folder: str) -> pd.DataFrame:
    df_summary =  load_all_data("articles", folder)
    df_full = load_all_data("full", folder, "parquet")
    df_full['text'] = df_full['text'].apply(clean_html)
    df = df_summary.merge(df_full, on='article_id')
    df = df[df['text'] != '']
    duplicate_columns = df.drop(columns=['article_id']).columns
    df = df.drop_duplicates(subset=duplicate_columns)
    df['publishDate'] = pd.to_datetime(df['publishDate'], format='mixed')
    return df


def _test() -> None:
    df = load_news_data("../../data/finlighten_news/")
    print(df.head().to_string())



if __name__ == "__main__":
    _test()


