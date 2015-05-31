package cs276.pa4;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Skeleton code for the implementation of a Cosine Similarity Scorer in Task 1.
 */
public class CosineSimilarityScorer extends AScorer {

	public CosineSimilarityScorer(Map<String,Double> idfs) {
		super(idfs);
	}

	/////////////// Weights //////////////////
	double urlweight = .7;
	double titleweight = 1;
	double bodyweight = .7;
	double headerweight = 1;
	double anchorweight = .6;

	double smoothingBodyLength = 500; // Smoothing factor when the body length is 0.
	//////////////////////////////////////////

	public double getNetScore(Map<String, Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery, Document d) {
		formatQuery(tfQuery);
		Map<String, Double> tfDocument = formatDocument(q.words,tfs);
		return dotProduct(tfQuery,tfDocument);
	}
	
	private void formatQuery(Map<String,Double> tfQuery){
		queryIdfWeighting(tfQuery);
	}
	
	@Override
	public Map<String, Double> formatDocument(List<String> terms, Map<String, Map<String, Double>> tfs){
		HashMap<String,Double> combinedVector = new HashMap<String,Double>();
		for(String word: terms){
			double score = 0;
			for(String type: TFTYPES){
				double weight = getWeight(type);
				Map<String,Double> sig = tfs.get(type);
				if(sig.containsKey(word)){
					score = score + sig.get(word)*weight;
				}
			}
			combinedVector.put(word,new Double(score));
		}
		return combinedVector;
	}
	
	private double getWeight(String type){
		if(type.equals("url")) return urlweight;
		if(type.equals("title")) return titleweight;
		if(type.equals("body")) return bodyweight;
		if(type.equals("header")) return headerweight;
		if(type.equals("anchor")) return anchorweight;
		return 0;
	}
	
	private double dotProduct(Map<String,Double> tfQuery, Map<String,Double> tfDocument){
		double score = 0.0;
		for(Map.Entry<String,Double> queryEntry : tfQuery.entrySet()){
			String key = queryEntry.getKey();
			double qScore = queryEntry.getValue();
			double dScore = 0.0;
			if(tfDocument.containsKey(key)){
				dScore = tfDocument.get(key);
			}
			score = score + qScore*dScore;
		}
		return score;
	}

	// Normalize the term frequencies. Note that we should give uniform normalization to all fields as discussed
	// in the assignment handout.
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
		double factor = d.body_length + smoothingBodyLength;
		for(Map.Entry<String,Map<String,Double>> entry : tfs.entrySet()){
			String key = entry.getKey();
			Map<String,Double> value = tfs.get(key);
			for(Map.Entry<String,Double> docEntry : value.entrySet()){
				double docKey = docEntry.getValue();
				docEntry.setValue(new Double(docKey/factor));
			}
		}
	}


	@Override
	public double getSimScore(Document d, Query q) {
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);

	    return getNetScore(tfs,q,tfQuery,d);
	}

}
