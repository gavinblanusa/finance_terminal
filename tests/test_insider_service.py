from datetime import date
import importlib
import sys
import types

import pytest
from insider_service import (
    _filing_document_url,
    fetch_insider_transactions_sec,
    parse_form4_xml,
    parse_recent_form4_filings,
)


FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerTradingSymbol>NVDA</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Jane Q Insider</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Financial Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-04-21</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>42.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-04-22</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>12</value></transactionShares>
        <transactionPricePerShare><value>50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def test_parse_form4_xml_normalizes_open_market_purchase_and_sale():
    rows = parse_form4_xml(
        FORM4_XML,
        sec_link="https://www.sec.gov/Archives/edgar/data/1/a/doc.xml",
        filing_date="2026-04-23",
    )

    assert len(rows) == 2
    assert rows[0] == {
        "date": date(2026, 4, 21),
        "filing_date": date(2026, 4, 23),
        "transaction": "Buy",
        "transaction_raw": "P",
        "shares": 100,
        "price": 42.5,
        "value": 4250,
        "name": "Jane Q Insider",
        "relationship": "Chief Financial Officer",
        "sec_link": "https://www.sec.gov/Archives/edgar/data/1/a/doc.xml",
        "source": "sec",
        "open_market": True,
        "is_officer": True,
        "is_director": True,
    }
    assert rows[1]["transaction"] == "Sale"
    assert rows[1]["transaction_raw"] == "S"
    assert rows[1]["value"] == 600
    assert rows[1]["open_market"] is True


def test_parse_form4_xml_skips_malformed_shares_and_keeps_missing_price():
    xml = """<ownershipDocument>
      <reportingOwner>
        <reportingOwnerId><rptOwnerName>Alex Director</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship>
          <isDirector>1</isDirector>
          <isOfficer>0</isOfficer>
        </reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-04-20</value></transactionDate>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>not-a-number</value></transactionShares>
            <transactionPricePerShare><value>10</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
          </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-04-21</value></transactionDate>
          <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>5</value></transactionShares>
            <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
          </transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    rows = parse_form4_xml(xml, sec_link="", filing_date="2026-04-22")

    assert len(rows) == 1
    assert rows[0]["shares"] == 5
    assert rows[0]["price"] is None
    assert rows[0]["value"] == 0
    assert rows[0]["relationship"] == "Director"
    assert rows[0]["open_market"] is False


def test_parse_recent_form4_filings_ignores_other_forms():
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "4", "4/A", "8-K"],
                "accessionNumber": ["000-1", "000-2", "000-3", "000-4"],
                "filingDate": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
                "reportDate": ["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31"],
                "primaryDocument": ["q.htm", "xslF345X05/form4.xml", "doc.xml", "eightk.htm"],
            }
        }
    }

    rows = parse_recent_form4_filings(submissions)

    assert [row["form"] for row in rows] == ["4", "4/A"]
    assert rows[0]["accession_number"] == "000-2"
    assert rows[0]["primary_document"] == "xslF345X05/form4.xml"


def test_filing_document_url_strips_sec_xsl_viewer_directory():
    url = _filing_document_url(
        "0000320193",
        "0001140361-26-017175",
        "xslF345X06/form4.xml",
    )

    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000114036126017175/form4.xml"


def test_fetch_insider_transactions_sec_uses_cache_roundtrip(monkeypatch, tmp_path):
    submissions = {
        "filings": {
            "recent": {
                "form": ["4"],
                "accessionNumber": ["0001045810-26-000001"],
                "filingDate": ["2026-04-23"],
                "reportDate": ["2026-04-22"],
                "primaryDocument": ["form4.xml"],
            }
        }
    }

    class FakeResponse:
        text = FORM4_XML

    monkeypatch.setattr("insider_service.CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        "insider_service.get_ticker_to_cik",
        lambda: {"NVDA": ("0001045810", "NVIDIA CORP")},
    )
    monkeypatch.setattr("insider_service._get_submissions_for_cik", lambda cik: submissions)
    monkeypatch.setattr("insider_service._sec_get", lambda url, headers, timeout=15: FakeResponse())

    first = fetch_insider_transactions_sec("nvda")

    monkeypatch.setattr(
        "insider_service._sec_get",
        lambda url, headers, timeout=15: pytest.fail("cache should satisfy second call"),
    )
    second = fetch_insider_transactions_sec("NVDA")

    assert first == second
    assert first[0]["source"] == "sec"
    assert first[0]["sec_link"].endswith("/form4.xml")


def _import_market_data_with_light_stubs(monkeypatch):
    sys.modules.pop("market_data", None)

    requests_stub = types.ModuleType("requests")
    requests_stub.get = lambda *args, **kwargs: None

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None

    sqlalchemy_stub = types.ModuleType("sqlalchemy")
    sqlalchemy_stub.and_ = lambda *args, **kwargs: None
    sqlalchemy_orm_stub = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm_stub.Session = object
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_data = lambda *args, **kwargs: (lambda fn: fn)

    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    pandas_stub.Series = object
    numpy_stub = types.ModuleType("numpy")

    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    monkeypatch.setitem(sys.modules, "yfinance", types.ModuleType("yfinance"))
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_stub)
    monkeypatch.setitem(sys.modules, "sqlalchemy.orm", sqlalchemy_orm_stub)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)

    return importlib.import_module("market_data")


def test_market_data_prefers_sec_rows(monkeypatch, tmp_path):
    market_data = _import_market_data_with_light_stubs(monkeypatch)
    sec_rows = [{"date": date(2026, 4, 21), "transaction": "Buy", "source": "sec"}]
    monkeypatch.setattr(market_data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(market_data, "fetch_insider_transactions_sec", lambda *args, **kwargs: sec_rows)
    monkeypatch.setattr(
        market_data.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("Finnhub should not be called when SEC has rows"),
    )

    assert market_data.fetch_insider_transactions("nvda") == sec_rows


def test_market_data_falls_back_to_finnhub_when_sec_has_no_rows(monkeypatch, tmp_path):
    market_data = _import_market_data_with_light_stubs(monkeypatch)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {
                        "date": "2026-04-21",
                        "transactionType": "P",
                        "shares": 7,
                        "USDValue": 350,
                        "name": "Backup Insider",
                        "relationship": "CEO",
                        "SECForm4Link": "https://sec.example/form4",
                    }
                ]
            }

    monkeypatch.setattr(market_data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(market_data, "FINNHUB_API_KEY", "token")
    monkeypatch.setattr(market_data, "fetch_insider_transactions_sec", lambda *args, **kwargs: [])
    monkeypatch.setattr(market_data.requests, "get", lambda *args, **kwargs: FakeResponse())

    rows = market_data.fetch_insider_transactions("nvda")

    assert rows == [
        {
            "date": date(2026, 4, 21),
            "transaction": "Buy",
            "transaction_raw": "P",
            "shares": 7,
            "value": 350,
            "name": "Backup Insider",
            "relationship": "CEO",
            "sec_link": "https://sec.example/form4",
            "source": "finnhub",
            "filing_date": date(2026, 4, 21),
            "price": None,
            "open_market": True,
            "is_officer": False,
            "is_director": False,
        }
    ]
