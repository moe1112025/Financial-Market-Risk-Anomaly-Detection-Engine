# 💹 Financial Market Risk & Anomaly Detection Engine

## Overview

The **Financial Market Risk & Anomaly Detection Engine** is a Python-based machine learning application designed to demonstrate the use of **unsupervised anomaly detection** for financial transaction monitoring.

The system uses an **Isolation Forest** model to identify transactions that exhibit behavior different from the patterns represented in a simulated financial transaction dataset.

An interactive **Streamlit dashboard** provides a monitoring interface where transaction parameters can be evaluated and converted into an anomaly-based risk verdict.

## Project Purpose

Financial monitoring systems need to identify unusual transaction behavior without necessarily relying on previously labeled examples of every possible fraud pattern.

This project demonstrates an unsupervised machine learning approach in which the model learns the structure of simulated transaction behavior and identifies observations that appear anomalous within that feature space.

The system focuses on **anomaly detection and risk signaling**, rather than attempting to prove that a transaction is definitively fraudulent.

## Core Functionality

The application evaluates four transaction and account-level parameters:

* **Transaction Amount ($)**
* **Transaction Velocity — Transactions in the Last Hour**
* **Account Age (Days)**
* **Geographic Distance from Home Location (km)**

These parameters are represented as machine learning features and passed to the trained anomaly detection model.

The system provides:

* Transaction telemetry visualization
* Anomaly classification
* Anomaly score calculation
* High-risk alert generation
* Normal transaction status reporting
* Manual security-review recommendation for anomalous transactions

## Machine Learning Implementation

The core detection engine uses Scikit-learn's **Isolation Forest** algorithm.

Isolation Forest is an unsupervised learning technique designed to identify observations that are easier to isolate from the rest of the dataset.

The model is trained using simulated transaction behavior generated with NumPy distributions representing different types of financial and account activity.

The implementation uses a contamination parameter of `0.05`, representing the expected proportion of anomalous observations within the simulated training data.

## Detection Workflow

The system follows this conceptual pipeline:

**Synthetic Transaction Data → Feature Representation → Isolation Forest Training → Transaction Input → Anomaly Prediction → Decision Function → Risk Verdict**

For an evaluated transaction, the model produces:

* `1` for an observation considered normal
* `-1` for an observation considered anomalous

The model's `decision_function()` is also used to obtain an anomaly score. In this implementation, lower scores indicate observations that are more anomalous relative to the learned data distribution.

## Risk Evaluation

When an anomalous transaction is identified, the application generates a **High Risk** security alert and recommends that the transaction be flagged for manual review.

For observations classified as normal, the dashboard reports the transaction as a legitimate-looking transaction within the learned behavioral distribution.

It is important to distinguish this anomaly verdict from a confirmed fraud determination: **anomaly detection identifies unusual behavior; it does not independently establish that fraud has occurred.**

## Interactive Monitoring Dashboard

The Streamlit interface provides an interactive environment for transaction evaluation.

Users can modify:

* Transaction amount
* Transaction velocity
* Account age
* Geographic distance

The selected values are converted into a Pandas DataFrame and passed directly into the trained Isolation Forest model for inference.

The dashboard then displays the resulting transaction data and security verdict.

## Technical Highlights

### Machine Learning

* Unsupervised anomaly detection
* Isolation Forest
* Behavioral outlier identification
* Decision-function-based anomaly scoring
* Contamination-based anomaly modeling

### Data Engineering

* NumPy synthetic data generation
* Multiple statistical distributions for simulated transaction behavior
* Pandas-based feature representation
* Four-dimensional transaction feature space

### Application Layer

* Streamlit interactive dashboard
* Dynamic transaction parameter controls
* Real-time model inference
* Risk verdict visualization
* Security alert generation

## Technology Stack

**Programming Language:** Python

**Machine Learning:** Scikit-learn

**Algorithm:** Isolation Forest

**Data Processing:** Pandas, NumPy

**Application Framework:** Streamlit

## Project Value

This project demonstrates how **unsupervised machine learning can be integrated into a financial security monitoring workflow** to identify unusual transaction behavior.

It provides a foundation for exploring more advanced financial anomaly detection systems involving real transaction histories, feature engineering, temporal behavioral analysis, adaptive thresholds, supervised fraud classification, and production-grade monitoring pipelines.

The current implementation intentionally uses **synthetic transaction data** as a controlled environment for demonstrating the anomaly detection architecture and model inference workflow.

**Moe Htet Ar Kar (Phoe Cho)**
