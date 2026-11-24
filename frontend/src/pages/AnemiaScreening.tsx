import { useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import { Upload, X, AlertTriangle, Info, CheckCircle, Loader2 } from "lucide-react";
import { api } from "../lib/api";

const MAX_SIZE_MB = 12;
// Android camera/gallery sometimes reports an empty MIME type for JPEG photos,
// so we also check the file extension as a fallback.
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"];
const ALLOWED_EXTENSIONS = /\.(jpe?g|png|webp|heic|heif)$/i;

interface UploadState {
  file: File | null;
  preview: string | null;
  error: string | null;
}

function emptyUpload(): UploadState {
  return { file: null, preview: null, error: null };
}

interface ScreeningResult {
  model: string;
  roi_detected: boolean;
  roi_label: string;
  roi_confidence: number;
  roi_box_xyxy: number[];
  anemia_probability: number;
  estimated_hemoglobin_g_dl: number;
  is_anemic_flag: boolean;
  heads_agree: boolean;
  risk_tier: "Low" | "Moderate" | "High";
  disclaimer: string;
}

interface NailScreeningResult {
  nail_model: string;
  nail_anemia_probability: number;
  nail_is_anemic: boolean;
  risk_tier: "Low" | "Moderate" | "High";
  record_id: number;
}

interface ScreeningResults {
  eyelid: ScreeningResult;
  nail: NailScreeningResult;
}

export default function AnemiaScreening() {
  const { t } = useTranslation();
  const [consented, setConsented] = useState(false);
  const [eyelid, setEyelid] = useState<UploadState>(emptyUpload());
  const [nail, setNail] = useState<UploadState>(emptyUpload());
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [result, setResult] = useState<ScreeningResults | null>(null);

  const eyelidRef = useRef<HTMLInputElement>(null);
  const nailRef = useRef<HTMLInputElement>(null);

  function handleFile(
    e: React.ChangeEvent<HTMLInputElement>,
    setter: (s: UploadState) => void
  ) {
    const file = e.target.files?.[0];
    if (!file) return;
    // Android gallery/camera apps sometimes report an empty MIME type for JPEG;
    // fall back to checking the file extension before rejecting the file.
    const typeOk = ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTENSIONS.test(file.name);
    if (!typeOk) {
      setter({ file: null, preview: null, error: t("forms.upload.type_limit", { types: "JPEG, PNG, WebP, HEIC" }) });
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setter({ file: null, preview: null, error: t("forms.upload.size_limit", { max_size: MAX_SIZE_MB }) });
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      setter({ file, preview: ev.target?.result as string, error: null });
    };
    reader.readAsDataURL(file);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setApiError(null);

    if (!eyelid.file) {
      setApiError(t("screening.missing_eyelid"));
      return;
    }
    if (!nail.file) {
      setApiError(t("screening.missing_nail"));
      return;
    }

    setLoading(true);
    try {
      const eyelidForm = new FormData();
      eyelidForm.append("file", eyelid.file);
      const nailForm = new FormData();
      nailForm.append("file", nail.file);

      const eyelidResult = await api.postForm<ScreeningResult>(
        "/api/screenings/analyze",
        eyelidForm
      );
      const nailResult = await api.postForm<NailScreeningResult>(
        "/api/screenings/analyze-nail",
        nailForm
      );
      setResult({ eyelid: eyelidResult, nail: nailResult });
    } catch (err) {
      setApiError(
        err instanceof Error
          ? err.message
          : "Network error. Unable to reach the screening service. Please check your connection and try again."
      );
    } finally {
      setLoading(false);
    }
  }

  if (result) {
    const eyelidResult = result.eyelid;
    const nailResult = result.nail;
    const isUncertain = !eyelidResult.roi_detected || !eyelidResult.heads_agree;
    const tierColors = {
      Low: "bg-emerald-50 text-emerald-800 border-emerald-200",
      Moderate: "bg-amber-50 text-amber-800 border-amber-200",
      High: "bg-rose-50 text-rose-800 border-rose-200",
    }[eyelidResult.risk_tier] || "bg-pink/10 text-deep border-pink/30";
    const nailTierColors = {
      Low: "bg-emerald-50 text-emerald-800 border-emerald-200",
      Moderate: "bg-amber-50 text-amber-800 border-amber-200",
      High: "bg-rose-50 text-rose-800 border-rose-200",
    }[nailResult.risk_tier] || "bg-pink/10 text-deep border-pink/30";

    return (
      <div className="page-container py-12 max-w-lg">
        <div className="bg-white rounded-xl border border-border shadow-soft p-8">
          <div className="text-center mb-6">
            <CheckCircle size={40} className="mx-auto text-sage mb-3" />
            <h2 className="font-serif text-section text-ink">Screening Assessment</h2>
            <p className="text-xs text-ink/60 mt-1">Eyelid model: {eyelidResult.model}</p>
          </div>

          <div className="space-y-4 mb-6">
            <div className="flex items-center justify-between p-4 rounded-lg bg-blush border border-border">
              <span className="text-sm font-medium text-ink">Inner eyelid risk tier</span>
              <span className={`text-xs px-3 py-1 font-semibold rounded-full border ${tierColors}`}>
                {eyelidResult.risk_tier} Risk
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-4 rounded-lg bg-blush border border-border text-center">
                <span className="text-xs text-ink/60 block mb-1">Inner eyelid estimated hemoglobin</span>
                <span className="text-xl font-bold text-ink">
                  {eyelidResult.estimated_hemoglobin_g_dl.toFixed(1)}{" "}
                  <span className="text-xs font-normal text-ink/60">g/dL</span>
                </span>
              </div>

              <div className="p-4 rounded-lg bg-blush border border-border text-center">
                <span className="text-xs text-ink/60 block mb-1">Inner eyelid anemia probability</span>
                <span className="text-xl font-bold text-ink">
                  {(eyelidResult.anemia_probability * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-blush border border-border">
              <div className="flex items-center justify-between gap-3 mb-2">
                <span className="text-sm font-medium text-ink">Fingernail model estimate</span>
                <span className={`text-xs px-3 py-1 font-semibold rounded-full border ${nailTierColors}`}>
                  {nailResult.risk_tier} Risk
                </span>
              </div>
              <p className="text-sm text-ink">
                Anemia probability: {(nailResult.nail_anemia_probability * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-ink/60 mt-2">
                {t("screening.nail_validation_notice")}
              </p>
            </div>

            {isUncertain && (
              <div className="p-3.5 rounded-lg bg-cream border border-deep/20 flex items-start gap-2.5 text-xs text-deep">
                <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                <span>
                  {!eyelidResult.roi_detected
                    ? "The conjunctiva was not clearly segmented. We recommend retaking the photo in bright light or getting a clinical CBC blood test."
                    : "The probability model and hemoglobin estimator produced diverging assessments. We recommend retaking the photo or consulting a clinician for a CBC blood test."}
                </span>
              </div>
            )}

            <div className="p-3.5 rounded-lg bg-blush/60 border border-border text-xs text-ink/70 leading-relaxed">
              <p className="font-medium text-ink mb-1">Important Notice</p>
              <p>{eyelidResult.disclaimer}</p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              setResult(null);
              setEyelid(emptyUpload());
              setNail(emptyUpload());
              setConsented(false);
              setApiError(null);
            }}
            className="w-full py-2.5 bg-deep text-white text-sm font-medium rounded-lg hover:bg-pink transition-colors"
          >
            Start over
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container py-8 max-w-2xl">
      <div className="mb-8">
        <p className="text-xs font-semibold text-pink uppercase tracking-wider mb-2">{t("screening.title")}</p>
        <h1 className="text-display-md font-serif text-ink">{t("screening.title")}</h1>
        <p className="text-ink/60 mt-2">{t("screening.subtitle")}</p>
      </div>

      {/* Prototype notice */}
      <div className="bg-cream border border-deep/20 rounded-xl p-5 mb-6 flex items-start gap-3">
        <AlertTriangle size={18} className="text-deep mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-deep mb-1">{t("screening.limitation")}</p>
          <p className="text-sm text-deep/80 leading-relaxed">{t("screening.limitation_text")}</p>
        </div>
      </div>

      {/* What it explores */}
      <div className="bg-white rounded-xl border border-border shadow-soft p-6 mb-6">
        <h2 className="font-serif font-semibold text-ink mb-3">{t("screening.what_it_is")}</h2>
        <p className="text-sm text-ink/70 leading-relaxed">{t("screening.what_it_is_desc")}</p>
      </div>

      {/* Consent */}
      <div className="bg-white rounded-xl border border-border shadow-soft p-6 mb-6">
        <div className="flex items-start gap-2 mb-4">
          <Info size={16} className="text-deep mt-0.5 shrink-0" />
          <h2 className="font-serif font-semibold text-ink">{t("screening.consent_heading")}</h2>
        </div>
        <p className="text-sm text-ink/70 leading-relaxed mb-4">{t("screening.consent_text")}</p>
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={consented}
            onChange={(e) => setConsented(e.target.checked)}
            className="mt-0.5 w-4 h-4 accent-deep"
          />
          <span className="text-sm text-ink">
            I understand this is a prototype screening tool, and that a clinical CBC blood test is required for definitive medical diagnosis.
          </span>
        </label>
      </div>

      {/* Error message */}
      {apiError && (
        <div className="mb-6 p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3">
          <AlertTriangle size={18} className="shrink-0 mt-0.5 text-rose-600" />
          <div className="text-sm leading-relaxed">
            <p className="font-semibold mb-0.5">Screening Notice</p>
            <p>{apiError}</p>
          </div>
        </div>
      )}

      {/* Upload form */}
      <form onSubmit={handleSubmit} className={!consented ? "opacity-50 pointer-events-none" : ""}>
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <UploadZone
            label={t("screening.upload_eyelid")}
            guidance={t("screening.guidance_eyelid")}
            state={eyelid}
            inputRef={eyelidRef}
            capture="environment"
            onChange={(e) => {
              setApiError(null);
              handleFile(e, setEyelid);
            }}
            onRemove={() => {
              setApiError(null);
              setEyelid(emptyUpload());
            }}
          />
          <UploadZone
            label={t("screening.upload_nail")}
            guidance={t("screening.guidance_nail")}
            state={nail}
            inputRef={nailRef}
            capture="environment"
            onChange={(e) => {
              setApiError(null);
              handleFile(e, setNail);
            }}
            onRemove={() => {
              setApiError(null);
              setNail(emptyUpload());
            }}
          />
        </div>

        <p className="text-xs text-ink/60 mb-4">{t("screening.nail_validation_notice")}</p>

        <button
          type="submit"
          disabled={loading || !consented}
          className="w-full py-3 bg-deep text-white text-sm font-medium rounded-lg hover:bg-pink transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading && <Loader2 size={16} className="animate-spin" />}
          {loading ? t("screening.analyzing") : t("screening.submit")}
        </button>
      </form>
    </div>
  );
}

function UploadZone({
  label,
  guidance,
  state,
  inputRef,
  onChange,
  onRemove,
  capture,
}: {
  label: string;
  guidance: string;
  state: UploadState;
  inputRef: React.RefObject<HTMLInputElement>;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRemove: () => void;
  /** Pass "environment" to open the rear camera by default on mobile */
  capture?: "environment" | "user";
}) {
  const { t } = useTranslation();

  return (
    <div>
      <p className="text-sm font-medium text-ink mb-2">{label}</p>
      <p className="text-xs text-ink/50 mb-3 leading-relaxed">{guidance}</p>

      {state.preview ? (
        <div className="relative rounded-xl overflow-hidden border border-border">
          <img src={state.preview} alt="Upload preview" className="w-full h-40 object-cover" />
          <button
            type="button"
            onClick={onRemove}
            className="absolute top-2 right-2 p-1.5 rounded-full bg-ink/60 text-white hover:bg-ink transition-colors"
            aria-label={t("forms.upload.remove")}
          >
            <X size={14} />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="w-full h-36 rounded-xl border-2 border-dashed border-border hover:border-pink bg-blush hover:bg-rose transition-colors flex flex-col items-center justify-center gap-2 text-ink/50 hover:text-deep"
        >
          <Upload size={24} />
          <span className="text-xs">{t("forms.upload.drop_zone")}</span>
          <span className="text-xs text-deep font-medium">{t("forms.upload.browse")}</span>
        </button>
      )}

      {state.error && (
        <p className="text-xs text-red-600 mt-2">{state.error}</p>
      )}

      {/* accept includes image/* so Android camera roll is not filtered out;
          the HEIC extensions are added for iOS Safari which uses extension matching. */}
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.heic,.heif"
        capture={capture}
        onChange={onChange}
        className="hidden"
      />
    </div>
  );
}
