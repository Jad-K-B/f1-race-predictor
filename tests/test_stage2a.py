"""Integrity and leakage tests for the Stage 2A historical dataset."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_predictor.stage2a import (  # noqa: E402
    BINARY_TARGET_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FORBIDDEN_CURRENT_RACE_FEATURES,
    NUMERIC_FEATURES,
    OUTCOME_METADATA_COLUMNS,
    OUTPUT_COLUMNS,
    PREPROCESSING_CONTRACT,
    TARGET_COLUMNS,
    Stage2AConfig,
    build_stage2a_dataset,
    load_raw_tables,
    write_stage2a_outputs,
)


class Stage2ADatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_dir = PROJECT_ROOT / "data" / "raw"
        cls.config = Stage2AConfig()
        cls.tables = load_raw_tables(cls.raw_dir)
        cls.dataset = build_stage2a_dataset(
            cls.raw_dir, config=cls.config, tables=cls.tables
        )

    def test_data_integrity_and_unique_driver_race_rows(self) -> None:
        data = self.dataset
        self.assertEqual(set(data["year"].unique()), set(range(2014, 2026)))
        self.assertFalse(data.duplicated(["race_id", "driver_id"]).any())
        self.assertFalse(
            data[["race_id", "year", "round", "race_date", "driver_id", "constructor_id"]]
            .isna()
            .any()
            .any()
        )
        self.assertTrue(data[BINARY_TARGET_COLUMNS].isin([0, 1]).all().all())
        self.assertFalse(
            data[NUMERIC_FEATURES].isin([float("inf"), float("-inf")]).any().any()
        )
        self.assertEqual(len(FEATURE_COLUMNS), 147)
        constant_features = [
            column for column in FEATURE_COLUMNS if data[column].nunique(dropna=True) <= 1
        ]
        self.assertEqual(constant_features, [])

    def test_targets_match_official_results(self) -> None:
        official = self.tables["results"]
        official = official[official["year"].between(2014, 2025)].copy()
        official["expected_points"] = pd.to_numeric(
            official["points"], errors="coerce"
        ).fillna(0).gt(0).astype("int8")
        official["expected_podium"] = pd.to_numeric(
            official["positionNumber"], errors="coerce"
        ).le(3).astype("int8")
        official["expected_winner"] = pd.to_numeric(
            official["positionNumber"], errors="coerce"
        ).eq(1).astype("int8")
        check = self.dataset.merge(
            official[
                [
                    "raceId",
                    "driverId",
                    "expected_points",
                    "expected_podium",
                    "expected_winner",
                ]
            ],
            left_on=["race_id", "driver_id"],
            right_on=["raceId", "driverId"],
            how="left",
            validate="one_to_one",
        )
        self.assertTrue(check["points_finish"].eq(check["expected_points"]).all())
        self.assertTrue(check["podium_finish"].eq(check["expected_podium"]).all())
        self.assertTrue(check["race_winner"].eq(check["expected_winner"]).all())
        per_race = self.dataset.groupby("race_id")[["podium_finish", "race_winner"]].sum()
        self.assertTrue(per_race["podium_finish"].eq(3).all())
        self.assertTrue(per_race["race_winner"].eq(1).all())

        eligible = self.dataset[self.dataset["prediction_eligible"].eq(1)]
        expected_order = (
            eligible.groupby("race_id")["finish_order"].rank(method="first")
        )
        pd.testing.assert_series_equal(
            eligible["finish_order"].astype("float64"),
            expected_order,
            check_names=False,
            check_dtype=False,
        )
        self.assertTrue(
            eligible.groupby("race_id")["finish_order"].min().eq(1).all()
        )
        self.assertTrue(
            eligible.groupby("race_id")["finish_order"].max().eq(
                eligible.groupby("race_id").size()
            ).all()
        )

    def test_prediction_eligibility_is_explicitly_audited(self) -> None:
        expected_ineligible = {
            (917, "valtteri-bottas"),
            (917, "will-stevens"),
            (917, "roberto-merhi"),
            (1057, "nikita-mazepin"),
            (1059, "mick-schumacher"),
            (1094, "lance-stroll"),
        }
        ineligible = self.dataset[self.dataset["prediction_eligible"].eq(0)]
        actual = set(zip(ineligible["race_id"], ineligible["driver_id"]))
        self.assertEqual(actual, expected_ineligible)
        self.assertTrue(ineligible["finish_order"].isna().all())
        self.assertTrue(ineligible["eligibility_source"].str.startswith("https://").all())

        results = self.tables["results"]
        grid_keys = set(
            zip(self.tables["grid"]["raceId"], self.tables["grid"]["driverId"])
        )
        dns_with_grid = results[
            results["year"].between(2014, 2025)
            & results["positionText"].eq("DNS")
            & results[["raceId", "driverId"]].apply(tuple, axis=1).isin(grid_keys)
        ]
        checked = self.dataset.merge(
            dns_with_grid[["raceId", "driverId"]],
            left_on=["race_id", "driver_id"],
            right_on=["raceId", "driverId"],
            how="inner",
        )
        self.assertEqual(len(checked), 33)
        self.assertTrue(checked["prediction_eligible"].eq(1).all())

    def test_missing_grid_record_never_auto_excludes_driver(self) -> None:
        altered_tables = copy.deepcopy(self.tables)
        candidate = altered_tables["grid"].query("year == 2024").iloc[0]
        remove = altered_tables["grid"]["raceId"].eq(candidate["raceId"]) & altered_tables[
            "grid"
        ]["driverId"].eq(candidate["driverId"])
        altered_tables["grid"] = altered_tables["grid"].loc[~remove].copy()
        with self.assertRaisesRegex(ValueError, "explicit prediction-timestamp eligibility audit"):
            build_stage2a_dataset(
                self.raw_dir, config=self.config, tables=altered_tables
            )

    def test_dnfs_remain_ranked_but_unclassified(self) -> None:
        dnf = self.dataset[
            self.dataset["result_status"].eq("DNF")
            & self.dataset["prediction_eligible"].eq(1)
        ]
        self.assertGreater(len(dnf), 0)
        self.assertTrue(dnf["classified_finish_position"].isna().all())
        self.assertTrue(dnf["finish_order"].notna().all())

    def test_splits_are_chronological_and_race_grouped(self) -> None:
        data = self.dataset
        self.assertTrue(data.loc[data["split"].eq("train"), "year"].le(2022).all())
        self.assertTrue(data.loc[data["split"].eq("validation"), "year"].eq(2023).all())
        self.assertTrue(data.loc[data["split"].eq("test"), "year"].between(2024, 2025).all())
        self.assertTrue(data.groupby("race_id")["split"].nunique().eq(1).all())
        dated = data.assign(parsed_race_date=pd.to_datetime(data["race_date"]))
        dates = dated.groupby("split")["parsed_race_date"].agg(["min", "max"])
        self.assertLess(dates.loc["train", "max"], dates.loc["validation", "min"])
        self.assertLess(dates.loc["validation", "max"], dates.loc["test", "min"])

    def test_pre_race_standings_are_shifted_one_race(self) -> None:
        races = self.tables["races"]
        races = races[races["year"].between(2014, 2025)].sort_values(
            ["year", "round", "date", "id"]
        )
        race_order = races[["id", "year", "round"]].copy()
        race_order["next_race_id"] = race_order.groupby("year")["id"].shift(-1)

        driver_expected = self.tables["driver_standings"].merge(
            race_order[["id", "next_race_id"]],
            left_on="raceId",
            right_on="id",
            how="inner",
        )
        driver_expected = driver_expected.rename(
            columns={
                "next_race_id": "race_id",
                "driverId": "driver_id",
                "positionNumber": "expected_driver_position",
                "points": "expected_driver_points",
            }
        ).dropna(subset=["race_id"])
        driver_check = self.dataset.merge(
            driver_expected[
                [
                    "race_id",
                    "driver_id",
                    "expected_driver_position",
                    "expected_driver_points",
                ]
            ],
            on=["race_id", "driver_id"],
            how="left",
        )
        comparable = driver_check["expected_driver_points"].notna()
        self.assertTrue(
            driver_check.loc[comparable, "driver_championship_points_pre_race"].eq(
                driver_check.loc[comparable, "expected_driver_points"]
            ).all()
        )
        self.assertTrue(
            driver_check.loc[comparable, "driver_championship_position_pre_race"].eq(
                driver_check.loc[comparable, "expected_driver_position"]
            ).all()
        )

        constructor_expected = self.tables["constructor_standings"].merge(
            race_order[["id", "next_race_id"]],
            left_on="raceId",
            right_on="id",
            how="inner",
        )
        constructor_expected = constructor_expected.rename(
            columns={
                "next_race_id": "race_id",
                "constructorId": "constructor_id",
                "positionNumber": "expected_constructor_position",
                "points": "expected_constructor_points",
            }
        ).dropna(subset=["race_id"])
        constructor_check = self.dataset.merge(
            constructor_expected[
                [
                    "race_id",
                    "constructor_id",
                    "expected_constructor_position",
                    "expected_constructor_points",
                ]
            ],
            on=["race_id", "constructor_id"],
            how="left",
        )
        comparable = constructor_check["expected_constructor_points"].notna()
        self.assertTrue(
            constructor_check.loc[
                comparable, "constructor_championship_points_pre_race"
            ].eq(constructor_check.loc[comparable, "expected_constructor_points"]).all()
        )
        self.assertTrue(
            constructor_check.loc[
                comparable, "constructor_championship_position_pre_race"
            ].eq(constructor_check.loc[comparable, "expected_constructor_position"]).all()
        )

        round_one = self.dataset["round"].eq(1)
        self.assertTrue(
            self.dataset.loc[round_one, "driver_championship_points_pre_race"].eq(0).all()
        )
        self.assertTrue(
            self.dataset.loc[
                round_one, "constructor_championship_points_pre_race"
            ].eq(0).all()
        )
        self.assertTrue(
            self.dataset.loc[
                round_one, "driver_championship_position_pre_race"
            ].isna().all()
        )
        self.assertTrue(
            self.dataset.loc[
                round_one, "constructor_championship_position_pre_race"
            ].isna().all()
        )

    def test_final_grid_and_qualifying_source_precedence(self) -> None:
        grid = self.tables["grid"]
        grid = grid[grid["year"].between(2014, 2025)][
            [
                "raceId",
                "driverId",
                "positionNumber",
                "positionText",
                "qualificationPositionNumber",
            ]
        ].rename(
            columns={
                "raceId": "race_id",
                "driverId": "driver_id",
                "positionNumber": "expected_grid",
                "positionText": "expected_grid_text",
                "qualificationPositionNumber": "grid_qualifying",
            }
        )
        check = self.dataset.merge(
            grid, on=["race_id", "driver_id"], how="inner", validate="one_to_one"
        ).sort_values(["race_id", "driver_id"])
        pd.testing.assert_series_equal(
            check["final_grid_position"].reset_index(drop=True),
            pd.to_numeric(check["expected_grid"], errors="coerce").reset_index(drop=True),
            check_names=False,
        )
        self.assertTrue(
            check["pit_lane_start"].eq(check["expected_grid_text"].eq("PL").astype(int)).all()
        )

        qualifying = self.tables["qualifying"].sort_values(
            ["raceId", "driverId", "positionDisplayOrder"]
        ).drop_duplicates(["raceId", "driverId"])
        qualifying = qualifying[["raceId", "driverId", "positionNumber"]].rename(
            columns={
                "raceId": "race_id",
                "driverId": "driver_id",
                "positionNumber": "session_qualifying",
            }
        )
        results = self.tables["results"]
        results = results[results["year"].between(2014, 2025)][
            ["raceId", "driverId", "qualificationPositionNumber"]
        ].rename(
            columns={
                "raceId": "race_id",
                "driverId": "driver_id",
                "qualificationPositionNumber": "result_qualifying_fallback",
            }
        )
        expected = self.dataset[["race_id", "driver_id", "qualifying_position"]].merge(
            grid[["race_id", "driver_id", "grid_qualifying"]],
            on=["race_id", "driver_id"],
            how="left",
        ).merge(
            qualifying, on=["race_id", "driver_id"], how="left"
        ).merge(results, on=["race_id", "driver_id"], how="left")
        expected_position = (
            pd.to_numeric(expected["grid_qualifying"], errors="coerce")
            .combine_first(pd.to_numeric(expected["session_qualifying"], errors="coerce"))
            .combine_first(
                pd.to_numeric(expected["result_qualifying_fallback"], errors="coerce")
            )
        )
        pd.testing.assert_series_equal(
            expected["qualifying_position"].reset_index(drop=True),
            expected_position.reset_index(drop=True),
            check_names=False,
        )

    def test_driver_history_is_strictly_prior(self) -> None:
        first_2014 = self.dataset[
            self.dataset["year"].eq(2014) & self.dataset["round"].eq(1)
        ]
        official = self.tables["results"]
        for row in first_2014.itertuples(index=False):
            expected = int(
                ((official["driverId"] == row.driver_id) & (official["year"] < 2014)).sum()
            )
            self.assertEqual(int(row.driver_prior_entries), expected)

    def test_current_outcomes_do_not_change_current_features(self) -> None:
        race_id = int(
            self.tables["races"].query("year == 2024").sort_values("round").iloc[0]["id"]
        )
        altered_tables = copy.deepcopy(self.tables)
        affected = altered_tables["results"]["raceId"].eq(race_id)
        original_positions = altered_tables["results"].loc[
            affected, "positionNumber"
        ].to_numpy(copy=True)
        original_points = altered_tables["results"].loc[
            affected, "points"
        ].to_numpy(copy=True)
        altered_tables["results"].loc[affected, "positionNumber"] = original_positions[::-1]
        altered_tables["results"].loc[affected, "positionDisplayOrder"] = (
            altered_tables["results"].loc[affected, "positionDisplayOrder"].to_numpy()[::-1]
        )
        altered_tables["results"].loc[affected, "positionText"] = "DNF"
        altered_tables["results"].loc[affected, "points"] = original_points[::-1]
        altered_tables["results"].loc[affected, "positionsGained"] = 99
        altered = build_stage2a_dataset(
            self.raw_dir, config=self.config, tables=altered_tables
        )
        cutoff = self.dataset.loc[self.dataset["race_id"].eq(race_id), "race_date"].iloc[0]
        original_race = self.dataset[self.dataset["race_date"].le(cutoff)].sort_values(
            ["race_id", "driver_id"]
        )
        altered_race = altered[altered["race_date"].le(cutoff)].sort_values(
            ["race_id", "driver_id"]
        )
        pd.testing.assert_frame_equal(
            original_race[FEATURE_COLUMNS].reset_index(drop=True),
            altered_race[FEATURE_COLUMNS].reset_index(drop=True),
            check_dtype=False,
        )
        self.assertFalse(
            original_race["finish_order"].reset_index(drop=True).equals(
                altered_race["finish_order"].reset_index(drop=True)
            )
        )

    def test_feature_contract_excludes_current_race_outcomes(self) -> None:
        self.assertFalse(set(FEATURE_COLUMNS) & FORBIDDEN_CURRENT_RACE_FEATURES)
        self.assertFalse(
            set(FEATURE_COLUMNS)
            & {
                "race_id",
                "race_date",
                "split",
                *TARGET_COLUMNS,
                *OUTCOME_METADATA_COLUMNS,
            }
        )
        self.assertEqual(list(self.dataset.columns), OUTPUT_COLUMNS)
        self.assertEqual(PREPROCESSING_CONTRACT["fit_preprocessors_on"], "train")
        self.assertEqual(PREPROCESSING_CONTRACT["transform_only"], ["validation", "test"])

    def test_invalid_config_and_missing_raw_schema_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Stage2AConfig(validation_year=2024)
        broken_tables = dict(self.tables)
        broken_tables["results"] = broken_tables["results"].drop(columns=["points"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_stage2a_dataset(
                self.raw_dir, config=self.config, tables=broken_tables
            )

    def test_writer_is_deterministic_and_split_files_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_dir = Path(first)
            second_dir = Path(second)
            first_metadata = write_stage2a_outputs(
                self.dataset, self.raw_dir, first_dir, self.config
            )
            second_metadata = write_stage2a_outputs(
                self.dataset, self.raw_dir, second_dir, self.config
            )
            for filename in (
                "race_features.csv",
                "train.csv",
                "validation.csv",
                "test.csv",
                "feature_definitions.json",
                "eligibility_audit.json",
                "metadata.json",
            ):
                self.assertEqual(
                    (first_dir / filename).read_bytes(),
                    (second_dir / filename).read_bytes(),
                )
            self.assertEqual(
                first_metadata["dataset"]["sha256"],
                second_metadata["dataset"]["sha256"],
            )

            split_races = {}
            split_rows = 0
            for split in ("train", "validation", "test"):
                part = pd.read_csv(first_dir / f"{split}.csv", low_memory=False)
                split_rows += len(part)
                split_races[split] = set(part["race_id"])
                self.assertTrue(part["split"].eq(split).all())
            self.assertEqual(split_rows, len(self.dataset))
            self.assertFalse(split_races["train"] & split_races["validation"])
            self.assertFalse(split_races["train"] & split_races["test"])
            self.assertFalse(split_races["validation"] & split_races["test"])

            definitions = json.loads(
                (first_dir / "feature_definitions.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["name"] for item in definitions["features"]], FEATURE_COLUMNS
            )
            self.assertTrue(
                all(item["source"] and item["temporal_scope"] for item in definitions["features"])
            )
            self.assertEqual(
                definitions["preprocessing_contract"], PREPROCESSING_CONTRACT
            )


if __name__ == "__main__":
    unittest.main()
