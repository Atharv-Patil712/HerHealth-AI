import { useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import { Upload, X, AlertTriangle, Info, CheckCircle , Loader2} from "lucide-react";
import { api } from "../lib/api";

const MAX_SIZE_MB = 8;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

interface UploadState {
  file: File | null;
  preview: string | null;
  error: string | null;
}

interface EyelidResult {
  risk_tier: "Low" | "Moderate" | "High" | "Unspecified";
  anemia_probability: number;
}

interface NailResult {
  nail_is_anemic: boolean;
  nail_anemia_probability: number;
}

interface ScreeningResult {
  eye: EyelidResult;
  nail: NailResult;
}

function emptyUpload(): UploadState {
  return { file: null, preview: null, error: null };
}

function parseProbability(value: unknown): number | null {
  const probability =
    typeof value === "number"
      ? value
      : typeof value === "string" && value.trim() !== ""
        ? Number(value)
        : Number.NaN;

  return Number.isFinite(probability) && probability >= 0 && probability <= 1
    ? probability
    : null;
}

function describeResponseShape(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value !== "object") return typeof value;

  const keys = Object.keys(value);
  return keys.length > 0 ? `fields: ${keys.join(", ")}` : "empty object";
}

function parseEyelidResult(value: unknown): EyelidResult | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  if (!("anemia_probability" in value)) return null;

  const probability = parseProbability(value.anemia_probability);
  if (probability === null) return null;

  const rawTier = "risk_tier" in value ? value.risk_tier : null;
  const normalizedTier =
    typeof rawTier === "string" ? rawTier.trim().toLowerCase() : "";
  const riskTier =
    normalizedTier === "low"
      ? "Low"
      : normalizedTier === "moderate"
        ? "Moderate"
        : normalizedTier === "high"
          ? "High"
          : "Unspecified";

  return { risk_tier: riskTier, anemia_probability: probability };
}

function parseNailResult(value: unknown): NailResult | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  if (!("nail_is_anemic" in value) || !("nail_anemia_probability" in value)) {
    return null;
  }

  const probability = parseProbability(value.nail_anemia_probability);
  if (typeof value.nail_is_anemic !== "boolean" || probability === null) {
    return null;
  }

  return {
    nail_is_anemic: value.nail_is_anemic,
    nail_anemia_probability: probability,
  };
}

export default function AnemiaScreening() {
  const { t } = useTranslation();
  const [consented, setConsented] = useState(false);
  //const [eyelid, setEyelid] = useState<UploadState>(emptyUpload());
  //const [nail, setNail] = useState<UploadState>(emptyUpload());
  //const [submitted, setSubmitted] = useState(false);
  const [eyelid, setEyelid] = useState<UploadState>(emptyUpload());
  const [nail, setNail] = useState<UploadState>(emptyUpload());
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [result, setResult] = useState<ScreeningResult | null>(null);

  const eyelidRef = useRef<HTMLInputElement>(null);
  const nailRef = useRef<HTMLInputElement>(null);

  // function handleFile(
  //   e: React.ChangeEvent<HTMLInputElement>,
  //   setter: (s: UploadState) => void
  // ) {
  //   const file = e.target.files?.[0];
  //   if (!file) return;
  //   if (!ALLOWED_TYPES.includes(file.type)) {
  //     setter({ file: null, preview: null, error: t("forms.upload.type_limit", { types: "JPEG, PNG, WebP" }) });
  //     return;
  //   }
  //   if (file.size > MAX_SIZE_MB * 1024 * 1024) {
  //     setter({ file: null, preview: null, error: t("forms.upload.size_limit", { max_size: MAX_SIZE_MB }) });
  //     return;
  //   }
  //   const reader = new FileReader();
  //   reader.onload = (ev) => {
  //     setter({ file, preview: ev.target?.result as string, error: null });
  //   };
  //   reader.readAsDataURL(file);
  // }

  // function handleSubmit(e: React.FormEvent) {
  //   e.preventDefault();
  //   if (!eyelid.file && !nail.file) return;
  //   // No image is uploaded to any server. We just record intent.
  //   setSubmitted(true);
  // }

  // if (submitted) {
  //   return (
  //     <div className="page-container py-12 max-w-lg">
  //       <div className="bg-white rounded-xl border border-border shadow-soft p-10 text-center">
  //         <CheckCircle size={40} className="mx-auto text-sage mb-4" />
  //         <h2 className="font-serif text-section text-ink mb-3">Images received (locally)</h2>
  //         <p className="text-sm text-ink/70 leading-relaxed mb-6">
  //           Your images have been validated locally. No image content has been sent to any server or analyzed by any model.
  //           Image analysis is not available in this prototype. Please consult a clinician for any health concerns.
  //         </p>
  //         <button
  //           onClick={() => { setSubmitted(false); setEyelid(emptyUpload()); setNail(emptyUpload()); setConsented(false); }}
  //           className="px-5 py-2 bg-deep text-white text-sm rounded-lg hover:bg-pink transition-colors"
  //         >
  //           Start over
  //         </button>
  //       </div>
  //     </div>
  //   );
  // }

  function handleFile(
  e: React.ChangeEvent<HTMLInputElement>,
  setter: (s: UploadState) => void
) {
  const file = e.target.files?.[0];
  if (!file) return;
  if (!ALLOWED_TYPES.includes(file.type)) {
    setter({ file: null, preview: null, error: t("forms.upload.type_limit", { types: "JPEG, PNG, WebP" }) });
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
    if (!eyelid.file) {
      setApiError(t("screening.missing_eyelid"));
      return;
    }
    if (!nail.file) {
      setApiError(t("screening.missing_nail"));
      return;
    }

    setLoading(true);
    setApiError(null);

    try {
      const eyeForm = new FormData();
      eyeForm.append("file", eyelid.file);
      const nailForm = new FormData();
      nailForm.append("file", nail.file);

      const eyeResponse = await api.postForm<unknown>(
        "/api/screenings/analyze",
        eyeForm
      );
      const eyeData = parseEyelidResult(eyeResponse);
      if (eyeResponse === null) {
        throw new Error(
          "No inner-eyelid result was returned. Please retake a clear, well-lit close-up photo of your inner lower eyelid."
        );
      }
      if (!eyeData) {
        throw new Error(
          `Invalid inner-eyelid response (${describeResponseShape(eyeResponse)}). Please retry.`
        );
      }

      const nailResponse = await api.postForm<unknown>(
        "/api/screenings/analyze-nail",
        nailForm
      );
      const nailData = parseNailResult(nailResponse);
      if (nailResponse === null) {
        throw new Error(
          "No fingernail result was returned. Please retake a clear, well-lit photo of your fingernail."
        );
      }
      if (!nailData) {
        throw new Error(
          `Invalid fingernail response (${describeResponseShape(nailResponse)}). Please retry.`
        );
      }

      setResult({ eye: eyeData, nail: nailData });
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Could not connect to server.");
    } finally {
      setLoading(false);
    }
  }

  if (result) {
    return (
      <div className="page-container py-12 max-w-lg">
        <div className="bg-white rounded-xl border border-border shadow-soft p-8 text-center">
          <CheckCircle size={40} className="mx-auto text-sage mb-4" />
          <h2 className="font-serif text-section text-ink mb-4">Screening Analysis Complete</h2>

         

          <div className="mb-6 text-left">
            <p className="text-xs font-semibold text-deep uppercase mb-2">
              Inner Eyelid Result ({result.eye.risk_tier} Risk)
            </p>
            <div className="bg-blush p-3 rounded-xl border border-border">
              <p className="text-xs text-ink/60">Inner Eyelid Anemia Probability</p>
              <p className="text-lg font-bold text-deep">{(result.eye.anemia_probability * 100).toFixed(1)}%</p>
            </div>
          </div>

          
          <div className="mb-6 text-left">
            <p className="text-xs font-semibold text-deep uppercase mb-2">
              Fingernail Bed Result ({result.nail.nail_is_anemic ? "Anemia Risk Detected" : "Normal"})
            </p>
            <div className="bg-blush p-3 rounded-xl border border-border">
              <p className="text-xs text-ink/60">Nail Anemia Probability</p>
              <p className="text-lg font-bold text-deep">{(result.nail.nail_anemia_probability * 100).toFixed(1)}%</p>
              <p className="mt-2 text-xs text-ink/60">{t("screening.nail_validation_notice")}</p>
            </div>
          </div>

          <p className="text-xs text-ink/70 leading-relaxed mb-6">
            Educational wellness prototype only. A clinical CBC blood test is required for medical diagnosis.
          </p>
          <button
            onClick={() => { setResult(null); setApiError(null); setEyelid(emptyUpload()); setNail(emptyUpload()); setConsented(false); }}
            className="px-5 py-2 bg-deep text-white text-sm rounded-lg hover:bg-pink transition-colors"
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
            I understand these experimental estimates are not a diagnosis and that a clinical CBC blood test is required to confirm anemia.
          </span>
        </label>
      </div>

      {/* Upload form */}
      <div className="flex items-center justify-between gap-4 mb-6">
        <StepBadge number={1} label={t("screening.consent_step", "Step 1: Consent")} active={consented} done={consented} />
        <StepBadge number={2} label={t("screening.upload_step", "Step 2: Photos")} active={consented && (!eyelid.file || !nail.file)} done={!!eyelid.file && !!nail.file} />
        <StepBadge number={3} label={t("screening.submit_step", "Step 3: Send")} active={consented && !!eyelid.file && !!nail.file} done={!!result} />
      </div>
      {apiError && (
        <div role="alert" className="bg-red-50 border border-red-200 text-red-800 rounded-x1 p-4 mb-6 text-sm">
          {apiError}
        </div>
      )}
      <form onSubmit={handleSubmit} className={!consented ? "opacity-50 pointer-events-none" : ""}>
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <UploadZone
            label={t("screening.upload_eyelid")}
            guidance={t("screening.guidance_eyelid")}
            state={eyelid}
            inputRef={eyelidRef}
            onChange={(e) => handleFile(e, setEyelid)}
            onRemove={() => setEyelid(emptyUpload())}
          />
          <UploadZone
            label={t("screening.upload_nail")}
            guidance={t("screening.guidance_nail")}
            state={nail}
            inputRef={nailRef}
            onChange={(e) => handleFile(e, setNail)}
            onRemove={() => setNail(emptyUpload())}
          />
        </div>

        <button
       type="submit"
       disabled={!consented || loading}
       className="w-full py-3 bg-deep text-white text-sm font-medium rounded-lg hover:bg-pink transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
     >
       {loading ? <><Loader2 size={16} className="animate-spin" /> {t("screening.analyzing")}</> : t("screening.submit")}
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
}: {
  label: string;
  guidance: string;
  state: UploadState;
  inputRef: React.RefObject<HTMLInputElement>;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRemove: () => void;
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

      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_TYPES.join(",")}
        onChange={onChange}
        className="hidden"
      />
    </div>
  );
}

function StepBadge({
  number,
  label,
  active,
  done,
}: {
  number: number;
  label: string;
  active: boolean;
  done: boolean;
}) {
  //const SYMBOLS = { 1: "📝", 2: "📷", 3: "✅" };
  const SYMBOLS: Record<number, string> = { 1: "📝", 2: "📷", 3: "✅" };
  return (
    <div className="flex flex-col items-center gap-1.5 flex-1">
      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shadow-sm border ${
        done
          ? "bg-deep text-white border-deep"
          : active
          ? "bg-pink text-white border-pink"
          : "bg-white text-ink/50 border-border"
      }`}>
        {done ? "✓" : number}
      </div>
      <span className={`text-[10px] font-medium leading-tight text-center ${
        done ? "text-deep" : active ? "text-deep" : "text-ink/50"
      }`}>{label}</span>
      <span className="text-[10px] text-ink/30">{SYMBOLS[number]}</span>
    </div>
  );
}
