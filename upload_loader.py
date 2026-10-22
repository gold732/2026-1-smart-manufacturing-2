"""Read uploaded tabular data without depending on the Streamlit UI."""
from pathlib import Path
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException


class UploadError(ValueError):
    """An actionable error that can be shown to the uploader."""


def load_uploaded_data(uploaded_file):
    extension = Path(getattr(uploaded_file, "name", "")).suffix.lower()
    if extension not in {".csv", ".xlsx"}:
        raise UploadError("지원하지 않는 파일 형식입니다. CSV(.csv) 또는 Excel(.xlsx) 파일을 업로드해 주세요.")

    uploaded_file.seek(0)
    try:
        if extension == ".csv":
            data = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        else:
            data = pd.read_excel(uploaded_file, engine="openpyxl")
    except UnicodeDecodeError as exc:
        raise UploadError("CSV 문자 인코딩을 읽을 수 없습니다. UTF-8 형식으로 저장한 뒤 다시 업로드해 주세요.") from exc
    except pd.errors.EmptyDataError as exc:
        raise UploadError("파일이 비어 있거나 열 이름이 없습니다. 열 이름과 데이터 행을 포함해 주세요.") from exc
    except pd.errors.ParserError as exc:
        raise UploadError("CSV 행의 열 개수가 맞지 않거나 구분자·따옴표 형식이 잘못되었습니다. 쉼표로 구분된 CSV인지 확인해 주세요.") from exc
    except ImportError as exc:
        raise UploadError("Excel 읽기에 필요한 openpyxl이 설치되어 있지 않습니다. 관리자에게 의존성 설치를 요청해 주세요.") from exc
    except (BadZipFile, InvalidFileException, ParseError, EOFError, KeyError) as exc:
        raise UploadError("Excel 파일이 손상되었거나 실제 XLSX 형식이 아닙니다. Excel에서 .xlsx로 다시 저장해 주세요.") from exc
    except (ValueError, OSError) as exc:
        label = "Excel" if extension == ".xlsx" else "CSV"
        raise UploadError(f"{label} 파일을 읽을 수 없습니다. 파일 손상 또는 암호 보호 여부를 확인하고 다시 저장해 주세요.") from exc

    if data.empty or data.dropna(how="all").empty:
        raise UploadError("분석할 데이터 행이 없습니다. 열 이름 아래에 한 행 이상의 데이터를 포함해 주세요.")
    return data
