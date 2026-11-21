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
    "screening.consent_step": "Step 1: Consent",
    "screening.upload_step": "Step 2: Photos",
    "screening.submit_step": "Step 3: Send",
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

describe("AnemiaScreening", () => {
  beforeEach(() => {
    mockedPostForm.mockReset();
  });

  it("requires an inner-eyelid photo when only a fingernail photo is selected", async () => {
    const user = userEvent.setup();
    const { container } = render(<AnemiaScreening />);
    await user.click(screen.getByRole("checkbox"));

    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    const nailPhoto = new File(["image"], "nail.jpg", { type: "image/jpeg" });
    await user.upload(inputs.item(1), nailPhoto);
    await waitFor(() => {
      expect(screen.getByAltText("Upload preview")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(
      await screen.findByText("Please upload a clear photo of your inner lower eyelid.")
    ).toBeInTheDocument();
    expect(mockedPostForm).not.toHaveBeenCalled();
  });

  it("requires a fingernail photo when only an inner-eyelid photo is selected", async () => {
    const user = userEvent.setup();
    const { container } = render(<AnemiaScreening />);
    await user.click(screen.getByRole("checkbox"));

    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    const eyelidPhoto = new File(["image"], "eyelid.jpg", { type: "image/jpeg" });
    await user.upload(inputs.item(0), eyelidPhoto);
    await waitFor(() => {
      expect(screen.getByAltText("Upload preview")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(await screen.findByText("Please upload a clear photo of a fingernail."))
      .toBeInTheDocument();
    expect(mockedPostForm).not.toHaveBeenCalled();
  });

  it("asks for a retake when the eyelid API returns null", async () => {
    const user = userEvent.setup();
    const { container } = render(<AnemiaScreening />);
    mockedPostForm.mockResolvedValueOnce(null);

    await user.click(screen.getByRole("checkbox"));
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    const photo = new File(["image"], "photo.jpg", { type: "image/jpeg" });
    await user.upload(inputs.item(0), photo);
    await user.upload(inputs.item(1), photo);
    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "No inner-eyelid result was returned. Please retake a clear, well-lit close-up photo of your inner lower eyelid."
    );
    expect(screen.getByRole("heading", { name: "Anemia Screening" })).toBeInTheDocument();
    expect(screen.queryByText("Screening Analysis Complete")).not.toBeInTheDocument();
    expect(mockedPostForm).toHaveBeenCalledTimes(1);
  });

  it("shows a recoverable error when the fingernail API returns an invalid result", async () => {
    const user = userEvent.setup();
    const { container } = render(<AnemiaScreening />);
    mockedPostForm
      .mockResolvedValueOnce({
        anemia_probability: "0.37",
        risk_tier: "low",
      })
      .mockResolvedValueOnce({});

    await user.click(screen.getByRole("checkbox"));
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    const photo = new File(["image"], "photo.jpg", { type: "image/jpeg" });
    await user.upload(inputs.item(0), photo);
    await user.upload(inputs.item(1), photo);
    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Invalid fingernail response (empty object)."
    );
    expect(screen.getByRole("heading", { name: "Anemia Screening" })).toBeInTheDocument();
    expect(screen.queryByText("Screening Analysis Complete")).not.toBeInTheDocument();
    expect(mockedPostForm).toHaveBeenCalledTimes(2);
  });

  it("shows both screening results, including the inner-eyelid probability", async () => {
    const user = userEvent.setup();
    const { container } = render(<AnemiaScreening />);
    mockedPostForm
      .mockResolvedValueOnce({
        anemia_probability: 0.37,
        risk_tier: "Low",
      })
      .mockResolvedValueOnce({
        nail_anemia_probability: 0.61,
        nail_is_anemic: true,
      });

    await user.click(screen.getByRole("checkbox"));
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="file"]');
    const photo = new File(["image"], "photo.jpg", { type: "image/jpeg" });
    await user.upload(inputs.item(0), photo);
    await user.upload(inputs.item(1), photo);
    await user.click(screen.getByRole("button", { name: "Analyze both photos" }));

    expect(
      await screen.findByText("Inner Eyelid Anemia Probability")
    ).toBeInTheDocument();
    expect(screen.queryByText(/hemoglobin/i)).not.toBeInTheDocument();
    expect(screen.getByText("37.0%")).toBeInTheDocument();
    expect(screen.getByText("Nail Anemia Probability")).toBeInTheDocument();
    expect(screen.getByText("61.0%")).toBeInTheDocument();
    expect(mockedPostForm).toHaveBeenCalledTimes(2);
  });
});
