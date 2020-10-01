import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import fnmatch
import  numpy as np
import random
import math

import re
import csv
import csv 
from functools import partial
from numpy.testing import assert_almost_equal, assert_array_almost_equal


#
# RAS2LPS
#

class RAS2LPS(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "RAS2LPS" # TODO make this more human readable by adding spaces
    self.parent.categories = ["SlicerMorph.SlicerMorph Labs"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto (Seattle Children's), Sara Rolfe (UW), Murat Maga (UW)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module takes a directory of mesh files saved in RAS coordinates, and converts them to LPS coordinates. Convenient to convert the modules that are saved prior to change LPS in Feb 2020. See https://discourse.slicer.org/t/model-files-are-now-saved-in-lps-coordinate-system/10446
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
      This module was developed by Arthur Porto, Sara Rolfe and Murat Maga, through a NSF ABI Development grant, "An Integrated Platform for Retrieval, Visualization and Analysis of
      3D Morphology From Digital Biological Collections" (Award Numbers: 1759883 (Murat Maga), 1759637 (Adam Summers), 1759839 (Douglas Boyer)).
      https://nsf.gov/awardsearch/showAward?AWD_ID=1759883&HistoricalAwards=false
      """ # replace with organization, grant and thanks.

#
# RAS2LPSWidget
#

class RAS2LPSWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def onSelect(self):
    self.applyButton.enabled = bool (self.inputDirectory.currentPath and self.outputDirectory.currentPath)

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    self.inputDirectory=ctk.ctkPathLineEdit()
    self.inputDirectory.filters=ctk.ctkPathLineEdit.Dirs
    self.inputDirectory.setToolTip( "Select input directory" )
    parametersFormLayout.addRow("Input folder:", self.inputDirectory)

    self.outputDirectory=ctk.ctkPathLineEdit()
    self.outputDirectory.filters=ctk.ctkPathLineEdit.Dirs
    self.outputDirectory.setToolTip( "Select output directory")
    parametersFormLayout.addRow("Output folder: ", self.outputDirectory)


    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Convert."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.inputDirectory.connect('validInputChanged(bool)', self.onSelect)
    self.outputDirectory.connect('validInputChanged(bool)', self.onSelect)
    self.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass


  def onApplyButton(self):
    logic = RAS2LPSLogic()
    logic.run(self.inputDirectory.currentPath, self.outputDirectory.currentPath)

#
# RAS2LPSLogic
#

class RAS2LPSLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def run(self, inputDirectory, outputDirectory):
    extension = ".ply"
    for file in os.listdir(inputDirectory):
      if file.endswith(extension):
        filePath = os.path.join(inputDirectory,file)
        modelNode = slicer.modules.models.logic().AddModel(filePath, slicer.vtkMRMLStorageNode.CoordinateSystemRAS)
        modelNode.GetDisplayNode().SetVisibility(False)
        modelNode.GetStorageNode().SetCoordinateSystem(slicer.vtkMRMLStorageNode.CoordinateSystemLPS)
        fileName = os.path.basename(filePath)
        (baseName, ext) = os.path.splitext(fileName)
        outputFileName = baseName + '-LPS.ply'
        outputFilePath = os.path.join(outputDirectory, outputFileName)
        slicer.util.saveNode(modelNode, outputFilePath)
        slicer.mrmlScene.RemoveNode(modelNode)
    slicer.util.infoDisplay("Your file conversion is complete")      


class RAS2LPSTest(ScriptedLoadableModuleTest):
  """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
      """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
      """
    self.setUp()
    self.test_RAS2LPS1()

  def test_RAS2LPS1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
      tests should exercise the functionality of the logic with different inputs
      (both valid and invalid).  At higher levels your tests should emulate the
      way the user would interact with your code and confirm that it still works
      the way you intended.
      One of the most important features of the tests is that it should alert other
      developers when their changes will have an impact on the behavior of your
      module.  For example, if a developer removes a feature that you depend on,
      your test should break so they know that the feature is needed.
      """
    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767',
      checksums='SHA256:12d17fba4f2e1f1a843f0757366f28c3f3e1a8bb38836f0de2a32bb1cd476560')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = RAS2LPSLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
