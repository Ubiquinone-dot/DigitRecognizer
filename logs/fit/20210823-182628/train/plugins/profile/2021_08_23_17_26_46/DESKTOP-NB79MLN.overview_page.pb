?	??4?8?4@??4?8?4@!??4?8?4@	]D??T??]D??T??!]D??T??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??4?8?4@e?`TR'??A?7??d?4@Y?/?'??rEagerKernelExecute 0*????̌H@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ǘ????!ͥ??7@@)lxz?,C??1A??<@:Preprocessing2U
Iterator::Model::ParallelMapV29??v????!"??
z:@)9??v????1"??
z:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateΈ?????!z? =??2@)??0?*x?1w??W(@:Preprocessing2F
Iterator::ModelΈ?????!z? =??B@)Ǻ???v?1?'ނ?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Zd;??!?z??O@)??H?}m?1|?f?S@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_?Q?k?!????ղ@)_?Q?k?1????ղ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!d?ΙK?@)a2U0*?c?1d?ΙK?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'???????!m
Ę??5@)Ǻ???V?1?'ނ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9\D??T??I^???U?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e?`TR'??e?`TR'??!e?`TR'??      ??!       "      ??!       *      ??!       2	?7??d?4@?7??d?4@!?7??d?4@:      ??!       B      ??!       J	?/?'???/?'??!?/?'??R      ??!       Z	?/?'???/?'??!?/?'??b      ??!       JCPU_ONLYY\D??T??b q^???U?X@Y      Y@q*n????@"?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 