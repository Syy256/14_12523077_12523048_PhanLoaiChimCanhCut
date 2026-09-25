import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";


// ============================================================
// API
// ============================================================

const API =
  import.meta.env.VITE_API_URL || "http://localhost:8000";


// ============================================================
// INITIAL FORM
// ============================================================

const initialForm = {
  island: "Biscoe",
  culmen_length_mm: 45.5,
  culmen_depth_mm: 17.1,
  flipper_length_mm: 195,
  body_mass_g: 4000,
  sex: "MALE",
};


// ============================================================
// FORMAT METRIC
// ============================================================

function formatMetric(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "number") {
    return value.toFixed(2);
  }

  return value;
}


// ============================================================
// PREDICTION CARD
// ============================================================

function PredictionCard({ result, metrics }) {

  // ----------------------------------------------------------
  // Chưa có prediction
  // ----------------------------------------------------------

  if (!result) {

    return (
      <div className="prediction-card coral">

        <div className="card-top">

          <span className="model-index">
            01
          </span>

          <span className="model-name">
            LogisticRegression
          </span>

          <span className="model-dot" />

        </div>


        <div className="prediction-main">

          <div>

            <div className="prediction-label">
              PREDICTION
            </div>

            <h3>
              —
            </h3>

          </div>


          <div className="confidence">

            <strong>
              —
            </strong>

            <span>
              CONFIDENCE
            </span>

          </div>

        </div>


        <div className="metric-title">
          TEST METRICS

          <span>
            MODEL
          </span>
        </div>


        <div className="metrics">

          <div>
            <strong>
              {formatMetric(metrics?.test_accuracy)}
            </strong>

            <span>
              ACC
            </span>
          </div>


          <div>
            <strong>
              {formatMetric(metrics?.precision_macro)}
            </strong>

            <span>
              PREC
            </span>
          </div>


          <div>
            <strong>
              {formatMetric(metrics?.recall_macro)}
            </strong>

            <span>
              REC
            </span>
          </div>


          <div>
            <strong>
              {formatMetric(metrics?.f1_macro)}
            </strong>

            <span>
              F1
            </span>
          </div>

        </div>

      </div>
    );
  }


  // ----------------------------------------------------------
  // Prediction đã có
  // ----------------------------------------------------------

  const probabilities =
    result.probabilities || {};


  return (
    <div className="prediction-card coral">

      {/* ----------------------------------------------------
          MODEL HEADER
      ----------------------------------------------------- */}

      <div className="card-top">

        <span className="model-index">
          01
        </span>

        <span className="model-name">
          {result.model_name || "LogisticRegression"}
        </span>

        <span className="model-dot" />

      </div>


      {/* ----------------------------------------------------
          MAIN RESULT
      ----------------------------------------------------- */}

      <div className="prediction-main">

        <div>

          <div className="prediction-label">
            PREDICTION
          </div>

          <h3>
            {result.prediction}
          </h3>

        </div>


        <div className="confidence">

          <strong>
            {(result.confidence * 100).toFixed(1)}%
          </strong>

          <span>
            CONFIDENCE
          </span>

        </div>

      </div>


      {/* ----------------------------------------------------
          PROBABILITIES
      ----------------------------------------------------- */}

      <div className="probability-list">

        {Object.entries(probabilities).map(
          ([label, probability]) => (

            <div
              className="probability"
              key={label}
            >

              <span>
                {label}
              </span>


              <div className="bar">

                <i
                  style={{
                    width: `${probability * 100}%`,
                  }}
                />

              </div>


              <b>
                {(probability * 100).toFixed(1)}%
              </b>

            </div>

          )
        )}

      </div>


      {/* ----------------------------------------------------
          METRICS
      ----------------------------------------------------- */}

      <div className="metric-title">

        TEST METRICS

        <span>
          v{result.model_version || "1.0.0"}
        </span>

      </div>


      <div className="metrics">

        <div>

          <strong>
            {formatMetric(metrics?.test_accuracy)}
          </strong>

          <span>
            ACC
          </span>

        </div>


        <div>

          <strong>
            {formatMetric(metrics?.precision_macro)}
          </strong>

          <span>
            PREC
          </span>

        </div>


        <div>

          <strong>
            {formatMetric(metrics?.recall_macro)}
          </strong>

          <span>
            REC
          </span>

        </div>


        <div>

          <strong>
            {formatMetric(metrics?.f1_macro)}
          </strong>

          <span>
            F1
          </span>

        </div>

      </div>

    </div>
  );
}


// ============================================================
// APP
// ============================================================

function App() {

  const [form, setForm] =
    useState(initialForm);

  const [result, setResult] =
    useState(null);

  const [modelInfo, setModelInfo] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [apiStatus, setApiStatus] =
    useState("CHECKING");


  // ==========================================================
  // LOAD MODEL INFO
  // ==========================================================

  useEffect(() => {

    async function loadModelInfo() {

      try {

        const response = await fetch(
          `${API}/api/model-info`
        );


        if (!response.ok) {

          throw new Error(
            "Không thể lấy thông tin model"
          );

        }


        const data =
          await response.json();


        setModelInfo(data);

        setApiStatus("ONLINE");

      } catch (err) {

        console.error(
          "Model info error:",
          err
        );

        setApiStatus("OFFLINE");

      }

    }


    loadModelInfo();

  }, []);


  // ==========================================================
  // HANDLE FORM CHANGE
  // ==========================================================

  function handleChange(event) {

    const {
      name,
      value,
      type,
    } = event.target;


    setForm((current) => ({

      ...current,

      [name]:
        type === "number"
          ? Number(value)
          : value,

    }));

  }


  // ==========================================================
  // HANDLE PREDICTION
  // ==========================================================

  async function handlePredict(event) {

    event.preventDefault();


    setLoading(true);

    setError("");

    setResult(null);


    try {

      const response = await fetch(
        `${API}/api/predict`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify(form),
        }
      );


      const data =
        await response.json();


      // ------------------------------------------------------
      // Backend trả lỗi
      // ------------------------------------------------------

      if (!response.ok) {

        let errorMessage =
          "Prediction failed";


        if (typeof data.detail === "string") {

          errorMessage =
            data.detail;

        } else if (
          Array.isArray(data.detail)
        ) {

          errorMessage =
            data.detail
              .map(
                (item) =>
                  item.msg || "Invalid input"
              )
              .join(", ");

        }


        throw new Error(
          errorMessage
        );

      }


      // ------------------------------------------------------
      // Thành công
      // ------------------------------------------------------

      setResult(data);

      setApiStatus("ONLINE");

    } catch (err) {

      console.error(
        "Prediction error:",
        err
      );


      setError(
        err.message ||
        "Không thể kết nối Backend"
      );


      setApiStatus("OFFLINE");

    } finally {

      setLoading(false);

    }

  }


  // ==========================================================
  // METRICS
  // ==========================================================

  const metrics =
    modelInfo?.metrics_test || null;


  // ==========================================================
  // RENDER
  // ==========================================================

  return (

    <div className="app-shell">


      {/* =====================================================
          TOP BAR
      ====================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            P
          </div>

          <span>
            PENGUIN CLASSIFIER
          </span>

        </div>


        <div className="nav-status">

          <span className="status-dot" />

          API {apiStatus}


          <span className="nav-divider" />


          LOGISTIC REGRESSION

        </div>

      </header>


      {/* =====================================================
          HERO
      ====================================================== */}

      <section className="hero">

        <div>

          <p className="eyebrow">

            MACHINE LEARNING

            <span>
              / DEPLOYED MODEL
            </span>

          </p>


          <h1>

            Penguin

            <br />

            <em>
              Classifier
            </em>

          </h1>


          <p className="hero-note">

            Nhập các đặc trưng của chim cánh cụt
            để sử dụng mô hình Logistic Regression
            đã được huấn luyện và đánh giá trên
            bộ dữ liệu Palmer Penguins.

          </p>

        </div>


        <div className="hero-stamp">

          <strong>
            01
          </strong>

          DEPLOYED
          <br />
          MODEL

        </div>

      </section>


      {/* =====================================================
          WORKSPACE
      ====================================================== */}

      <main className="workspace">


        {/* ===================================================
            INPUT SECTION
        ==================================================== */}

        <div className="section-heading">

          <div>

            <span className="section-number">
              01
            </span>

            <h2>
              Penguin features
            </h2>

          </div>


          <span className="live-label">

            ● LIVE API

          </span>

        </div>


        {/* ===================================================
            FORM
        ==================================================== */}

        <form
          className="form-panel"
          onSubmit={handlePredict}
        >

          <div className="form-grid">


            {/* ------------------------------------------------
                ISLAND
            ------------------------------------------------- */}

            <div className="field">

              <label className="field-label">

                ISLAND

                <small>
                  Location
                </small>

              </label>


              <select
                name="island"
                value={form.island}
                onChange={handleChange}
              >

                <option value="Biscoe">
                  Biscoe
                </option>

                <option value="Dream">
                  Dream
                </option>

                <option value="Torgersen">
                  Torgersen
                </option>

              </select>

            </div>


            {/* ------------------------------------------------
                CULMEN LENGTH
            ------------------------------------------------- */}

            <div className="field">

              <label className="field-label">

                CULMEN LENGTH

                <small>
                  Bill length
                </small>

              </label>


              <div className="number-wrap">

                <input
                  type="number"
                  step="0.1"
                  min="0"
                  name="culmen_length_mm"
                  value={form.culmen_length_mm}
                  onChange={handleChange}
                  required
                />


                <span>
                  mm
                </span>

              </div>

            </div>


            {/* ------------------------------------------------
                CULMEN DEPTH
            ------------------------------------------------- */}

            <div className="field">

              <label className="field-label">

                CULMEN DEPTH

                <small>
                  Bill depth
                </small>

              </label>


              <div className="number-wrap">

                <input
                  type="number"
                  step="0.1"
                  min="0"
                  name="culmen_depth_mm"
                  value={form.culmen_depth_mm}
                  onChange={handleChange}
                  required
                />


                <span>
                  mm
                </span>

              </div>

            </div>


            {/* ------------------------------------------------
                FLIPPER LENGTH
            ------------------------------------------------- */}

            <div className="field">

              <label className="field-label">

                FLIPPER LENGTH

                <small>
                  Flipper length
                </small>

              </label>


              <div className="number-wrap">

                <input
                  type="number"
                  step="1"
                  min="0"
                  name="flipper_length_mm"
                  value={form.flipper_length_mm}
                  onChange={handleChange}
                  required
                />


                <span>
                  mm
                </span>

              </div>

            </div>


            {/* ------------------------------------------------
                BODY MASS
            ------------------------------------------------- */}

            <div className="field">

              <label className="field-label">

                BODY MASS

                <small>
                  Body weight
                </small>

              </label>


              <div className="number-wrap">

                <input
                  type="number"
                  step="1"
                  min="0"
                  name="body_mass_g"
                  value={form.body_mass_g}
                  onChange={handleChange}
                  required
                />


                <span>
                  g
                </span>

              </div>

            </div>


            {/* ------------------------------------------------
                SEX
            ------------------------------------------------- */}

            <div className="field">

              <label className="field-label">

                SEX

                <small>
                  Biological sex
                </small>

              </label>


              <select
                name="sex"
                value={form.sex}
                onChange={handleChange}
              >

                <option value="MALE">
                  MALE
                </option>

                <option value="FEMALE">
                  FEMALE
                </option>

              </select>

            </div>

          </div>


          {/* =================================================
              FORM FOOTER
          ================================================== */}

          <div className="form-footer">

            <span className="dataset-tag">
              6 INPUT FEATURES
            </span>


            <button
              type="submit"
              disabled={loading}
            >

              {loading
                ? "PREDICTING..."
                : "PREDICT →"}

            </button>

          </div>

        </form>


        {/* ===================================================
            RESULT SECTION
        ==================================================== */}

        <section className="results-section">


          <div className="section-heading">

            <div>

              <span className="section-number">
                02
              </span>

              <h2>
                Prediction result
              </h2>

            </div>


            <span className="live-label">
              DEPLOYED MODEL
            </span>

          </div>


          {/* -------------------------------------------------
              ERROR
          -------------------------------------------------- */}

          {error && (

            <div className="error-box">

              {error}

            </div>

          )}


          {/* -------------------------------------------------
              RESULT
          -------------------------------------------------- */}

          {!error && (

            <div className="result-grid">

              <PredictionCard
                result={result}
                metrics={metrics}
              />

            </div>

          )}

        </section>

      </main>


      {/* =====================================================
          FOOTER
      ====================================================== */}

      <footer>

        <span>
          PALMER ARCHIPELAGO PENGUIN DATASET
        </span>


        <span>

          MODEL: LOGISTIC REGRESSION ·{" "}

          {result?.model_version ||
            modelInfo?.model_version ||
            "1.0.0"}

        </span>

      </footer>

    </div>

  );
}


export default App;

createRoot(document.getElementById("root")).render(<App />);

