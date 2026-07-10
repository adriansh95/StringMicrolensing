from tasks.kde_label_task import KDELabelTask
from tasks.summary_table_task import SummaryTableTask
from tasks.efficiency_task import EfficiencyTask
from tasks.analyze_backgrounds_task import AnalyzeBackgroundsTask
from tasks.bin_objects_task import BinObjectsTask
from tasks.event_rate_task import EventRateTask
from tasks.effective_monitoring_time_task import (
    EffectiveMonitoringTimeTask
)
from tasks.good_windows_task import GoodWindowsTask
from tasks.good_windows_by_duration_task import (
    GoodWindowsByDurationTask
)
from tasks.sample_good_windows_task import SampleGoodWindowsTask
from tasks.random_sample_sources_task import RandomSampleSourcesTask
from tasks.event_stats_task import EventStatsTask
from tasks.simulated_event_stats_task import SimulatedEventStatsTask
from tasks.lens_lightcurves_task import LensLightcurvesTask
try:
    from tasks.rubin_sim_emt_task import RubinSimEMTTask
except ModuleNotFoundError as e:
    if e.name == "rubin_sim":
        RubinSimEMTTask = None
    else:
        raise

__all__ = [
    "KDELabelTask",
    "SummaryTableTask",
    "EfficiencyTask",
    "AnalyzeBackgroundsTask",
    "BinObjectsTask",
    "EventRateTask",
    "EffectiveMonitoringTimeTask",
    "GoodWindowsTask",
    "GoodWindowsByDurationTask",
    "SampleGoodWindowsTask",
    "RandomSampleSourcesTask",
    "EventStatsTask",
    "SimulatedEventStatsTask",
    "LensLightcurvesTask"
]

if RubinSimEMTTask is not None:
    __all__.append("RubinSimEMTTask")
