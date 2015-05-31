package cs276.pa4;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Skeleton code for the implementation of a BM25 Scorer in Task 2.
 */
public class BM25Scorer extends AScorer {
	Map<Query,Map<String, Document>> queryDict; // query -> url -> document

	public BM25Scorer(Map<String,Double> idfs, Map<Query,Map<String, Document>> queryDict) {
		super(idfs);
		this.queryDict = queryDict;
		this.calcAverageLengths();
	}


	/////////////// Weights /////////////////
	double urlweight = .7;
	double titleweight  = 1;
	double bodyweight = .7;
	double headerweight = 1;
	double anchorweight = .6;

	/////// BM25 specific weights ///////////
	double burl=.75;
	double btitle=.5;
	double bheader=.3;
	double bbody=.2;
	double banchor=.75;

	double k1=1;
	double pageRankLambda=.5;
	double pageRankLambdaPrime=1;
	//////////////////////////////////////////

	/////// BM25 data structures - feel free to modify ///////

	Map<Document,Map<String,Double>> lengths; // Document -> field -> length
	Map<String,Double> avgLengths;  // field name -> average length
	Map<Document,Double> pagerankScores; // Document -> pagerank score

	//////////////////////////////////////////

	// Set up average lengths for bm25, also handles pagerank
	public void calcAverageLengths() {
		//document -> field -> length
		lengths = new HashMap<Document,Map<String,Double>>();
		//field -> average length
		avgLengths = new HashMap<String,Double>();
		//document -> log(pagerank)
		pagerankScores = new HashMap<Document,Double>();
		
		for(Map.Entry<Query,Map<String, Document>> entry : queryDict.entrySet()){
			Map<String, Document> urlDocsMap = entry.getValue();
			for(Map.Entry<String,Document> entry2: urlDocsMap.entrySet()){
				Document d = entry2.getValue();
				if(!lengths.containsKey(d)){
					populateLengths(d,lengths);
					pagerankScores.put(d,new Double(computeVf(d.page_rank)));
				}
			}
		}
		
		//normalize avgLengths
		for(String tfType : this.TFTYPES) {
			double score = 0;
			for(Map.Entry<Document,Map<String,Double>> entry: lengths.entrySet()){
				Map<String,Double> value = entry.getValue();
				score = score + value.get(tfType);
			}
			score = score/lengths.size();
			avgLengths.put(tfType,new Double(score));
		}

	}
	
	private double computeVf(double page_rank){
		return Math.log(page_rank + pageRankLambdaPrime);
	}
	
	private double urlLength(Document d){
		double size = 0;
		size = d.url.split("\\W+").length;
		return size;
	}
	
	private double titleLength(Document d){
		double size = 0;
		String title = d.title;
		if(title != null){
			size = title.split("\\s+").length;
		}
		return size;
	}
	
	private double bodyLength(Document d){
		double size = 0;
		Map<String, List<Integer>> body_hits = d.body_hits;
		if(body_hits != null && body_hits.size() > 0){
			for(Map.Entry<String, List<Integer>> entry : body_hits.entrySet()){
				size = size + entry.getValue().size();
			}
			size = size/body_hits.size();
		}
		return size;
	}
	
	private double headerLength(Document d){
		double size = 0;
		List<String> headers = d.headers;
		if(headers != null && headers.size() > 0){
			for(String term: headers){
				size = size + term.split("\\s+").length;
			}
			size = size/headers.size();
		}
		return size;
	}
	
	private double anchorLength(Document d){
		double size = 0;
		Map<String, Integer> anchors = d.anchors;
		if(anchors != null && anchors.size() > 0){
			for(Map.Entry<String,Integer> entry: anchors.entrySet()){
				size = size + entry.getValue();
			}
			size = size/anchors.size();
		}
		return size;
	}
	
	private double getLength(Document d, String tfType){
		double size = 0.0;
		if(tfType.equals("url")){
			size = urlLength(d);
		}
		if(tfType.equals("title")){
			size = titleLength(d);
		}
		if(tfType.equals("body")){
			size = bodyLength(d);
		}
		if(tfType.equals("header")){
			size = headerLength(d);
		}
		if(tfType.equals("anchor")){
			size = anchorLength(d);
		}
		return size;
	}
	
	private double getWeight(String tfType,boolean isB){
		if(tfType.equals("url")){
			if(isB){
				return burl;
			}else{
				return urlweight;
			}
		}
		if(tfType.equals("title")){
			if(isB){
				return btitle;
			}else{
				return titleweight;
			}
		}
		if(tfType.equals("body")){
			if(isB){
				return bbody;
			}else{
				return bodyweight;
			}
		}
		if(tfType.equals("header")){
			if(isB){
				return bheader;
			}else{
				return headerweight;
			}
		}
		if(tfType.equals("anchor")){
			if(isB){
				return banchor;
			}else{
				return anchorweight;
			}
		}
		return 0;
	}
	
	private void populateLengths(Document d, Map<Document,Map<String,Double>> lengths){
		Map<String,Double> fields = new HashMap<String,Double>();
		for (String tfType : this.TFTYPES) {
			double size = getLength(d,tfType);
			fields.put(tfType,size);
		}
		lengths.put(d,fields);
	}

	////////////////////////////////////


	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d) {
		double score = 0.0;
		//Need to sum all of the terms now
		Map<String,Double> combinedScore = formatDocument(q.words,tfs);
		double pg = .001;
		if(pagerankScores.containsKey(d)){
			pg = pagerankScores.get(d);
		}
		for(String term: q.words){
			double w = combinedScore.get(term);
			double idf = getTermIDF(term);
			double num = w*idf;
			double den = w+k1;
			score = score + (num/den);
		}
		score = score + pageRankLambda*pg;
		return score;
	}

	//do bm25 normalization
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
		for(Map.Entry<String,Map<String,Double>> entry : tfs.entrySet()){
			//Field
			String key = entry.getKey();
			double len = getLength(d,key);
			double avgLen = avgLengths.get(key);
			//QueryTerm -> Value
			Map<String,Double> value = tfs.get(key);
			for(Map.Entry<String,Double> docEntry : value.entrySet()){
				double ftnum = docEntry.getValue();
				double ftden = 1 + getWeight(key,true)*((len/avgLen) - 1);
				double weightedFT = getWeight(key,false)*(ftnum/ftden);
				docEntry.setValue(new Double(weightedFT));
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
