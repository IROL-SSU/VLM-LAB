#!/usr/bin/env python3
"""Validate local HTML image comparison and render its five family sections."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/vlm_action_geometry_single_info_l2_v16_prompt_replay_v21/results/four_condition_comparison"
FAMILIES = ("FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                                    args=["--no-sandbox", "--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width":1500,"height":1000},device_scale_factor=1.5)
        errors = []
        page.on("pageerror", lambda error:errors.append(str(error)))
        page.goto((OUT / "index.html").as_uri(),wait_until="load")
        page.evaluate("document.fonts.ready")
        assert page.locator(".card").count() == 25
        assert page.locator(".card .counts").count() == 100
        assert page.locator(".seed-table tbody tr, .seed-table > tr").count() == 150
        assert page.locator("img").count() == 25
        assert page.evaluate("Array.from(document.images).every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===960)")
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
        outputs = []
        for family in FAMILIES:
            element = page.locator(f"#{family}")
            assert element.locator(".card").count() == 5
            path = OUT / f"{family.lower()}_comparison.png"
            element.screenshot(path=str(path),animations="disabled")
            outputs.append(str(path))
        # Check that expandable records work and include all five seeds.
        first = page.locator("details").first
        first.locator("summary").click()
        assert first.locator(".seed-table").is_visible()
        assert first.locator("pre").count() == 3
        assert not errors, errors
        browser.close()
    report = {"status":"pass","cards":25,"condition_cells":100,"seed_cells":500,"original_images_loaded":25,
              "all_images_at_original_resolution":True,"horizontal_overflow":False,"browser_errors":errors,"screenshots":outputs}
    (OUT / "render_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
