from src.core.database import Base
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import relationship


class ModelVersion(Base):
    __tablename__ = "ModelVersion"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(100), nullable=False)
    target = Column(String(100), nullable=False)
    trained_at = Column(DateTime(timezone=True), nullable=False)
    train_end_date = Column(Date, nullable=False)
    n_features = Column(Integer, nullable=False)
    train_auc = Column(Float, nullable=True)
    test_auc = Column(Float, nullable=True)
    train_r2 = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    feature_list = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False, server_default=text("false"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    predictions = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    __tablename__ = "Prediction"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "model_version_id",
            "prediction_date",
            name="uq_prediction_company_model_date",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("Company.company_id"), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey("ModelVersion.id"), nullable=False, index=True)
    prediction_date = Column(Date, nullable=False, index=True)
    predicted_price_5 = Column(Float, nullable=True)
    prob_direction_up = Column(Float, nullable=False)
    prob_momentum_5d = Column(Float, nullable=False)
    predicted_magnitude = Column(Float, nullable=False)
    ensemble_score = Column(Float, nullable=False)
    signal = Column(String(20), nullable=False)
    close_at_signal = Column(Float, nullable=False)
    actual_close_21d = Column(Float, nullable=True)
    actual_return_21d = Column(Float, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    company = relationship("Company")
    model_version = relationship("ModelVersion", back_populates="predictions")
