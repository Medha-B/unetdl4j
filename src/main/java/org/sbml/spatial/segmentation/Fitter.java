package org.sbml.spatial.segmentation;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;

public class Fitter {
	private DataSetIterator imageDataSetIterator;
	private int numEpochs;

	public Fitter(DataSetIterator imageDataSetIterator, int numEpochs) {
		this.imageDataSetIterator = imageDataSetIterator;
		this.numEpochs = numEpochs;
	}

	public ComputationGraph FitModel() {
		/*
		 * Map<Integer, Double> learningScheduleMap = new HashMap<>();
		 * learningScheduleMap.put(0, 0.00005); learningScheduleMap.put(200, 0.00001);
		 * learningScheduleMap.put(600, 0.000005); learningScheduleMap.put(800,
		 * 0.0000001); //learningScheduleMap.put(1000, 0.00001);
		 */

		UIServer uiServer = UIServer.getInstance();
		StatsStorage ss = new InMemoryStatsStorage();
		uiServer.attach(ss);

		ComputationGraph model = UNet.builder().updater(new Adam(1e-4)).build().init();
//        ComputationGraph model  = UNet.builder().updater(new Adam(new MapSchedule(ScheduleType.ITERATION, learningScheduleMap))).build().init();

		// For single channel input
		// ZooModel unet = UNet.builder().build();
		// unet.setInputShape(new int[][]{{1, 128, 128}});
		// ComputationGraph model = (ComputationGraph) unet.init();

		model.addListeners(new ScoreIterationListener(), new StatsListener(ss));
		model.fit(imageDataSetIterator, numEpochs);

		return model;

	}

}
