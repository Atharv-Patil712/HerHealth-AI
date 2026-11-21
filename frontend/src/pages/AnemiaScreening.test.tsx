import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api } from "../lib/api";
import AnemiaScreening from "./AnemiaScreening";

vi.mock("react-i18next", () => {
  const translations: Record<string, string> = {
    "forms.upload.type_limit": "Allowed types: {{types}}",
    "forms.upload.size_limit": "Image must be under {{max_size}}MB",
    "forms.upload.remove": "Remove",
    "forms.upload.drop_zone": "Drag and drop image here",
    "forms.upload.browse": "Browse files",
    "screening.title": "Anemia Screening",
    "screening.subtitle": "Image-assisted screening.",
    "screening.limitation": "Important limitation",
    "screening.limitation_text": "Experimental results are not a diagnosis.",
    "screening.what_it_is": "What this feature explores",
    "screening.what_it_is_desc": "A screening prototype.",
    "screening.consent_heading": "Your consent",
    "screening.consent_text": "Images are processed by HerHealth.",
    "screening.nail_validation_notice": "The nail model does not verify image anatomy.",
    "screening.upload_eyelid": "Upload inner eyelid photo",
    "screening.upload_nail": "Upload fingernail photo",
    "screening.guidance_eyelid": "Use natural light.",
    "screening.guidance_nail": "Natural nails only.",
    "screening.missing_eyelid": "Please upload a clear photo of your inner lower eyelid.",
    "screening.missing_nail": "Please upload a clear photo of a fingernail.",
    "screening.submit": "Analyze both photos",
    "screening.analyzing": "Analyzing both photos...",
  };
  return {
    useTranslation: () => ({
      t: (key: string) => translations[key] ?? key,
    }),
  };
});

vi.mock("../lib/api", () => ({
  api: { postForm: vi.fn() },
}));

const mockedPostForm = vi.mocked(api.postForm);

describe("AnemiaScreening uploads", () => {
  beforeEach(() => {
    mockedPostForm.mockReset();
  });

  it("asks for an eyelid photo when none was selected", async () => {
    const user = userEvent.setup();
    render(<AnemiaScreening />);

    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(await screen.findByText("Please upload a clear photo of your inner lower eyelid."))
      .toBeInTheDocument();
    expect(mockedPostForm).not.toHaveBeenCalled();
  });

  it("asks for a fingernail photo when only the eyelid photo was selected", async () => {
    const user = userEvent.setup();
    const { container } = render(<AnemiaScreening />);
    await user.click(screen.getByRole("checkbox"));

    const eyelidInput = container.querySelector<HTMLInputElement>('input[type="file"]');
    if (!eyelidInput) {
      throw new Error("The eyelid image input was not rendered.");
    }
    await user.upload(
      eyelidInput,
      new File(["image"], "eyelid.jpg", { type: "image/jpeg" })
    );
    await waitFor(() => {
      expect(screen.getByAltText("Upload preview")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(await screen.findByText("Please upload a clear photo of a fingernail."))
      .toBeInTheDocument();
    expect(mockedPostForm).not.toHaveBeenCalled();
  });

  it("shows the inner-eyelid probability and fingernail estimate", async () => {
    const user = userEvent.setup();
    const { container } = render(<AnemiaScreening />);
    mockedPostForm
      .mockResolvedValueOnce({
        model: "test-eyelid-model",
        roi_detected: true,
        roi_label: "conjunctiva",
        roi_confidence: 0.9,
        roi_box_xyxy: [0, 0, 1, 1],
        anemia_probability: 0.37,
        estimated_hemoglobin_g_dl: 12.5,
        is_anemic_flag: false,
        heads_agree: true,
        risk_tier: "Low",
        disclaimer: "A CBC test is required.",
      })
      .mockResolvedValueOnce({
        nail_model: "test-nail-model",
        nail_anemia_probability: 0.61,
        nail_is_anemic: true,
        risk_tier: "Moderate",
        record_id: 2,
      });

    await user.click(screen.getByRole("checkbox"));
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    if (inputs.length !== 2) {
      throw new Error("Both image inputs should be rendered.");
    }
    const photo = new File(["image"], "photo.jpg", { type: "image/jpeg" });
    await user.upload(inputs.item(0), photo);
    await user.upload(inputs.item(1), photo);
    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(await screen.findByText("Inner eyelid anemia probability")).toBeInTheDocument();
    expect(screen.getByText("37.0%")).toBeInTheDocument();
    expect(screen.getByText("Fingernail model estimate")).toBeInTheDocument();
    expect(screen.getByText("Anemia probability: 61.0%")).toBeInTheDocument();
  });
});
