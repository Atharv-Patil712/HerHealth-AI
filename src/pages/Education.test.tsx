import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { api } from "../lib/api";
import Education from "./Education";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("../lib/api", () => ({
  api: { get: vi.fn() },
}));

const mockedGet = vi.mocked(api.get);

describe("Education page", () => {
  beforeEach(() => {
    mockedGet.mockReset();
    vi.stubGlobal("localStorage", {
      getItem: () => null,
      setItem: vi.fn(),
    });
  });

  it("loads an article when its URL is opened directly", async () => {
    mockedGet
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce({
        id: "anemia-iron-deficiency",
        title: "Understanding anemia",
        category: "Anemia",
        excerpt: "An article excerpt.",
        read_time_minutes: 5,
        content: "## Article content\n\nDetails about the article.",
        references: [],
      });

    render(
      <MemoryRouter initialEntries={["/education/anemia-iron-deficiency"]}>
        <Routes>
          <Route path="/education" element={<Education />} />
          <Route path="/education/:articleId" element={<Education />} />
        </Routes>
      </MemoryRouter>
    );

    expect(
      await screen.findByRole("heading", { name: "Understanding anemia" })
    ).toBeInTheDocument();
    expect(screen.getByText("Details about the article.")).toBeInTheDocument();
    expect(mockedGet).toHaveBeenCalledWith(
      "/api/education/anemia-iron-deficiency"
    );
  });
});
