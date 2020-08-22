package org.sbml.spatial.segmentation;

import java.io.File;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//This class is for training the UNet model on any number of images and then saving it.
public class UnetTrainAndSave {

	private static final Logger log = LoggerFactory.getLogger(UnetTrainAndSave.class);

	public static void main(String[] args) {
		try {
			int batchSize = 10;
			// String home = System.getProperty("user.home");
			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			File rootDir = new File(dataPath + File.separator + "small_dataset");
			PreProcess prep = new PreProcess(rootDir, batchSize);
			DataSetIterator imageDataSetIterator = prep.dataProcessed();

			int numEpochs = 100;

			log.info("*****TRAIN MODEL******");
			Fitter fit = new Fitter(imageDataSetIterator, numEpochs);
			ComputationGraph model = fit.FitModel();
			// log.warn(model.summary());

			log.info("*****SAVE MODEL******");

			// Location for saving the model
			// File locationTosave = new File(home + File.separator + "unetSave.zip");
			File locationTosave = new File(directory + File.separator + "saveModel" + File.separator + "unetSave.zip");
			boolean saveUpdater = false;
			// ModelSerializer needs Model name, Location of saving the model and
			// saveUpdater.
			ModelSerializer.writeModel(model, locationTosave, saveUpdater);

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}
	}
}
