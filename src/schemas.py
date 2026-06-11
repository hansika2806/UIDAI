"""
Pandera schemas for raw cleaned UIDAI datasets.
Enforces type consistency, non-negative bounds, and required columns.
"""

import pandera.pandas as pa
from pandera.pandas import DataFrameModel, Field
from pandera.typing import Series
import pandas as pd


class EnrolmentSchema(DataFrameModel):
    date: Series[str] = Field(nullable=False, coerce=True)
    state: Series[str] = Field(nullable=False)
    district: Series[str] = Field(nullable=True)
    pincode: Series[str] = Field(coerce=True)
    age_0_5: Series[int] = Field(ge=0, coerce=True)
    age_5_17: Series[int] = Field(ge=0, coerce=True)
    age_18_greater: Series[int] = Field(ge=0, coerce=True)

    class Config:
        coerce = True
        strict = False


class DemographicUpdateSchema(DataFrameModel):
    date: Series[str] = Field(nullable=False, coerce=True)
    state: Series[str] = Field(nullable=False)
    district: Series[str] = Field(nullable=True)
    pincode: Series[str] = Field(coerce=True)
    demo_age_5_17: Series[int] = Field(ge=0, coerce=True)
    demo_age_17_: Series[int] = Field(ge=0, coerce=True)

    class Config:
        coerce = True
        strict = False


class BiometricUpdateSchema(DataFrameModel):
    date: Series[str] = Field(nullable=False, coerce=True)
    state: Series[str] = Field(nullable=False)
    district: Series[str] = Field(nullable=True)
    pincode: Series[str] = Field(coerce=True)
    bio_age_5_17: Series[int] = Field(ge=0, coerce=True)
    bio_age_17_: Series[int] = Field(ge=0, coerce=True)

    class Config:
        coerce = True
        strict = False
