# -*- coding: utf-8 -*-
"""Dialog to perform spatial autocorrelation tests."""

import io
import numpy as np

from qgis.PyQt.QtCore import Qt, QCoreApplication, QVariant
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGroupBox,
    QGridLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QTextEdit,
    QPushButton,
    QTabWidget,
    QDialogButtonBox,
    QWidget,
    QMessageBox,
    QHBoxLayout,
    QSizePolicy
)
from qgis.PyQt.QtGui import QPixmap

from qgis.core import (
    QgsVectorLayer,
    QgsField,
    QgsFields,
    QgsFeature,
    QgsProject,
    QgsWkbTypes,
)

class SpatialAutocorrelationDialog(QDialog):
    def __init__(self, layer, weight_data, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.weight_data = weight_data
        self.w = weight_data.get('weights')
        self.summary = weight_data.get('summary', '')
        self.setWindowTitle(self.tr('Spatial Autocorrelation Test'))

        main_layout = QVBoxLayout(self)
        param_tabs = QTabWidget()
        var_widget = self._create_variables_widget()
        param_tabs.addTab(var_widget, self.tr('Variable Setting'))
        param_tabs.addTab(self._create_report_widget(), self.tr('Weight Report'))
        param_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        main_layout.addWidget(param_tabs)

        self._setup_test_tabs(main_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)
        self.resize(int(self.sizeHint().width() * 1.5), self.sizeHint().height())

    # Section 1
    def _create_variables_widget(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.addWidget(QLabel(self.tr('First Variable')), 0, 0)
        self.firstCombo = QComboBox()
        for f in self.layer.fields():
            self.firstCombo.addItem(f.name())
        layout.addWidget(self.firstCombo, 0, 1)
        self.secondCheck = QCheckBox(self.tr('Second Variable'))
        self.secondCheck.stateChanged.connect(self._toggle_second)
        layout.addWidget(self.secondCheck, 1, 0)
        self.secondCombo = QComboBox()
        self.secondCombo.setEnabled(False)
        for f in self.layer.fields():
            self.secondCombo.addItem(f.name())
        layout.addWidget(self.secondCombo, 1, 1)
        layout.addWidget(QLabel(self.tr('Permutations')), 2, 0)
        self.permSpin = QSpinBox()
        self.permSpin.setRange(1, 99999)
        self.permSpin.setValue(999)
        layout.addWidget(self.permSpin, 2, 1)
        return widget

    def _toggle_second(self, state):
        self.secondCombo.setEnabled(state == Qt.Checked)

    # Section 2
    def _create_report_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.reportText = QTextEdit()
        self.reportText.setReadOnly(True)
        self.reportText.setPlainText(self.summary)
        self.reportText.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        layout.addWidget(self.reportText)
        return widget

    # Section 3
    def _setup_test_tabs(self, main_layout):
        box = QGroupBox(self.tr('Tests'))
        layout = QVBoxLayout(box)
        self.tabs = QTabWidget()
        self.moran_widgets = self._create_test_tab('Moran')
        self.tabs.addTab(self.moran_widgets[0], self.tr("Moran's I"))
        self.geary_widgets = self._create_test_tab('Geary')
        self.tabs.addTab(self.geary_widgets[0], self.tr("Geary's C"))
        layout.addWidget(self.tabs)
        main_layout.addWidget(box)

    def _create_test_tab(self, name):
        widget = QWidget()
        v = QVBoxLayout(widget)
        h = QHBoxLayout()
        run_btn = QPushButton(self.tr('Run'))
        add_cb = QCheckBox(self.tr('테스트 결과를 레이어로 추가'))
        h.addWidget(run_btn)
        h.addWidget(add_cb)
        h.addStretch()
        v.addLayout(h)
        text = QTextEdit()
        text.setReadOnly(True)
        plot = QLabel()
        plot.setMinimumHeight(300)
        plot.setAlignment(Qt.AlignCenter)
        plot.setScaledContents(True)
        v.addWidget(plot)
        v.addWidget(text)
        run_btn.clicked.connect(lambda: self.run_test(name, text, plot, add_cb))
        return widget, text, plot, add_cb

    def run_test(self, test_name, text_edit, plot_label, add_layer_cb):
        first = self.firstCombo.currentText()
        if not first:
            QMessageBox.warning(self, self.tr('Variables'), self.tr('First variable must be selected.'))
            return
        values1 = [feat[first] for feat in self.layer.getFeatures()]
        values2 = None
        if self.secondCheck.isChecked():
            second = self.secondCombo.currentText()
            values2 = [feat[second] for feat in self.layer.getFeatures()]
        perms = self.permSpin.value()
        try:
            from esda.moran import Moran, Moran_BV
            from esda.geary import Geary
        except Exception as e:
            QMessageBox.warning(self, self.tr('Dependency error'), str(e))
            return
        x = np.array(values1, dtype=float)
        if test_name == 'Moran':
            if values2 is None:
                res = Moran(x, self.w, permutations=perms)
                report = [
                    f"Moran's I: {res.I}",
                    f"Expectation: {res.EI}",
                    f"Std Dev: {res.seI_sim}",
                    f"P-value: {res.p_sim}"
                ]
                lag = self.w.sparse * x
            else:
                y = np.array(values2, dtype=float)
                res = Moran_BV(x, y, self.w, permutations=perms)
                report = [
                    f"Bivariate Moran's I: {res.I}",
                    f"P-value: {res.p_sim}"
                ]
                lag = self.w.sparse * y
            std_vals = (x - x.mean()) / x.std()
            lag_vals = np.asarray(lag).flatten()
            wy = (lag_vals - lag_vals.mean()) / lag_vals.std()
            text_edit.setPlainText('\n'.join(report))
            self._plot_scatter(std_vals, wy, res.I, plot_label)
            if add_layer_cb.isChecked():
                self._add_result_layer(lag_vals, std_vals, 'Moran')
        else:  # Geary
            if values2 is None:
                res = Geary(x, self.w, permutations=perms)
                report = [
                    f"Geary's C: {res.C}",
                    f"P-value: {res.p_sim}"
                ]
                lag = self.w.sparse * x
            else:
                res = Geary(x, self.w, permutations=perms)
                report = [
                    f"Geary's C: {res.C}",
                    f"P-value: {res.p_sim}",
                    self.tr('Bivariate Geary not available; second variable ignored.')
                ]
                lag = self.w.sparse * x
            std_vals = (x - x.mean()) / x.std()
            lag_vals = np.asarray(lag).flatten()
            wy = (lag_vals - lag_vals.mean()) / lag_vals.std()
            text_edit.setPlainText('\n'.join(report))
            self._plot_scatter(std_vals, wy, res.C, plot_label)
            if add_layer_cb.isChecked():
                self._add_result_layer(lag_vals, std_vals, 'Geary')

    def _plot_scatter(self, z, wy, slope, label):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            dpi = 100
            w = label.width() or 400
            h = label.height() or 300
            fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax.scatter(z, wy, s=10)
            xr = np.array([z.min(), z.max()])
            ax.plot(xr, slope * xr, 'r')
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
            ax.set_xlabel(self.tr('Standardized Value'))
            ax.set_ylabel(self.tr('Spatial Lag'))
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            label.setPixmap(pix)
        except Exception:
            pass

    def _add_result_layer(self, lag_vals, std_vals, test_name):
        fields = QgsFields(self.layer.fields())
        fields.append(QgsField('Spatial_Lag', QVariant.Double))
        fields.append(QgsField('STD', QVariant.Double))

        wkb = self.layer.wkbType()
        try:
            geom = QgsWkbTypes.displayString(wkb)
        except TypeError:
            geom = QgsWkbTypes.displayString(QgsWkbTypes.Type(wkb))

        if hasattr(self.layer, 'crs'):
            crs = self.layer.crs()
        else:
            crs = self.layer.sourceCrs()
        uri = f'{geom}?crs={crs.authid()}'

        if hasattr(self.layer, 'name'):
            base_name = self.layer.name()
        elif hasattr(self.layer, 'sourceName'):
            base_name = self.layer.sourceName()
        else:
            base_name = 'layer'
        if test_name == 'Moran':
            layer_name = f"{base_name} Moran's i"
        else:
            layer_name = f"{base_name} Geary's C"
        new_layer = QgsVectorLayer(uri, layer_name, 'memory')
        pr = new_layer.dataProvider()
        pr.addAttributes(fields)
        new_layer.updateFields()
        feats = []
        for feat, lag, std in zip(self.layer.getFeatures(), lag_vals, std_vals):
            nf = QgsFeature(new_layer.fields())
            nf.setGeometry(feat.geometry())
            nf.setAttributes(feat.attributes() + [float(lag), float(std)])
            feats.append(nf)
        pr.addFeatures(feats)
        QgsProject.instance().addMapLayer(new_layer)

    def tr(self, string):
        return QCoreApplication.translate('SpatialAutocorrelationDialog', string)