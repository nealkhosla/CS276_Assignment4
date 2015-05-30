package cs276.pa4;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Collections;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointwiseLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		// query -> documents list
		Map<Query,List<Document>> trainData = null;
		// query -> (url -> score)
		Map<String, Map<String, Double>> trainRels = null;
		try{
			trainData = Util.loadTrainData(train_data_file);
			trainRels = Util.loadRelData(train_rel_file);
		}catch(Exception e){
			System.err.println("Error loading training data");
		}
		for (Map.Entry<Query, List<Document>> entry : trainData.entrySet()){
			Query q = entry.getKey();
			List<Document> documents = entry.getValue();
			for(Document d : documents){
				double[] instance = FormatDocument.createInstanceVector(d, q, idfs, trainRels);
				Instance inst = new DenseInstance(1.0,instance);
				dataset.add(inst);
			}
		}
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		LinearRegression model = null;
		try{
			model = new LinearRegression();
			model.buildClassifier(dataset);
		}catch(Exception e){
			System.err.println("Error training PointwiseLearner LinearRegression model");
		}
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		// query -> documents list
		Map<Query,List<Document>> testData = null;
		try{
			testData = Util.loadTrainData(test_data_file);
		}catch(Exception e){
			System.err.println("Error loading training data");
		}
		
		Map<String, Map<String, Integer>> index_map = new HashMap<String, Map<String,Integer>>();
		
		int index = 0;
		for (Map.Entry<Query, List<Document>> entry : testData.entrySet()){
			Query q = entry.getKey();
			List<Document> documents = entry.getValue();
			for(Document d : documents){
				double[] instance = FormatDocument.createInstanceVector(d, q, idfs, null);
				Instance inst = new DenseInstance(1.0,instance);
				dataset.add(inst);
				Map<String,Integer> mp = new HashMap<String,Integer>(1);
				mp.put(d.url, new Integer(index));
				index_map.put(q.query,mp);
				index = index + 1;
			}
		}
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		TestFeatures tf = new TestFeatures();
		tf.features = dataset;
		tf.index_map = index_map;
		return tf;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		
		Map<String, List<String>> rankings = new HashMap<String, List<String>>();
		Instances test_dataset = tf.features;
		Map<String, Map<String,Integer>> index_map = tf.index_map;
		for (Map.Entry<String, Map<String,Integer>> entry1 : index_map.entrySet()){
			String query = entry1.getKey();
			Map<String,Integer> docMap = entry1.getValue();
			List<Pair<String,Double>> list = new ArrayList<Pair<String,Double>>();
			for (Map.Entry<String,Integer> entry2 : docMap.entrySet()){
				double prediction = Double.MIN_VALUE;
				String url = entry2.getKey();
				Integer index = entry2.getValue();
				try{
					prediction = model.classifyInstance(test_dataset.get(index.intValue()));
					//System.err.println("prediction: " + prediction);
				}catch(Exception e){
					System.err.println("Error classifying url " + url);
				}
				Pair<String,Double> p = new Pair<String,Double>(url,new Double(prediction));
				list.add(p);
			}
			sortList(list);
			rankings.put(query,convertList(list));
		}
		return rankings;
	}
	
	private void sortList(List<Pair<String,Double>> list){
		Collections.sort(list, new Comparator<Pair<String,Double>>(){
			@Override
		    public int compare(Pair<String,Double> p1, Pair<String,Double> p2) {
		        return p1.getSecond().compareTo(p2.getSecond());
		    }
		});
	}
	
	private List<String> convertList(List<Pair<String,Double>> list){
		List<String> rankings = new ArrayList<String>(list.size());
		for(int i = 0; i < list.size(); i++){
			rankings.add(list.get(i).getFirst());
		}
		return rankings;
	}

}