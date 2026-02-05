from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PySide6.QtCore import Qt

from msa_workbench.ui.widgets.mpl_canvas import MplCanvas
from msa_workbench.plotting import get_variability_chart, get_stddev_chart

class ChartsPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False) # Let the content define its size
        self.setFocusPolicy(Qt.NoFocus)

        charts_content = QWidget()
        self.setWidget(charts_content)
        
        self.charts_content_layout = QVBoxLayout(charts_content)
        self.charts_content_layout.setSpacing(20)
        self.charts_content_layout.setAlignment(Qt.AlignHCenter) # Center the plots horizontally
        
        self.variability_chart_canvas = MplCanvas(self)
        self.variability_chart_canvas.setFocusPolicy(Qt.NoFocus)

        self.stddev_chart_canvas = MplCanvas(self)
        self.stddev_chart_canvas.setFocusPolicy(Qt.NoFocus)

        self.charts_content_layout.addWidget(self.variability_chart_canvas)
        self.charts_content_layout.addWidget(self.stddev_chart_canvas)

    def update_charts(self, result):
        if result is None:
            return
        
        print("Updating charts with new size logic!")

        n_groups = len(result.chart_data.stddev) if result.chart_data and result.chart_data.stddev is not None else 10
        
        # Calculate width dynamically based on groups, with a minimum
        plot_width_px = max(900, n_groups * 75)
        
        # Calculate height based on 3:2 aspect ratio (width:height)
        plot_height_px = int(plot_width_px * (2 / 3))

        # Set the fixed size for the canvases
        self.variability_chart_canvas.setFixedSize(plot_width_px, plot_height_px)
        self.stddev_chart_canvas.setFixedSize(plot_width_px, plot_height_px)
        
        # Update matplotlib figure size in inches
        dpi = self.variability_chart_canvas.figure.dpi
        self.variability_chart_canvas.figure.set_size_inches(plot_width_px / dpi, plot_height_px / dpi, forward=True)
        self.stddev_chart_canvas.figure.set_size_inches(plot_width_px / dpi, plot_height_px / dpi, forward=True)

        self.variability_chart_canvas.figure.clear()
        ax_var = self.variability_chart_canvas.figure.add_subplot(111)
        get_variability_chart(result, ax_var)
        self.variability_chart_canvas.figure.tight_layout(pad=3.0)
        self.variability_chart_canvas.draw()

        self.stddev_chart_canvas.figure.clear()
        ax_std = self.stddev_chart_canvas.figure.add_subplot(111)
        get_stddev_chart(result, ax_std)
        self.stddev_chart_canvas.figure.tight_layout(pad=3.0)
        self.stddev_chart_canvas.draw()

