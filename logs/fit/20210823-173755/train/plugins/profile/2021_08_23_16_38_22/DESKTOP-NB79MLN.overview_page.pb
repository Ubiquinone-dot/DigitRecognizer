?	?,C??5@?,C??5@!?,C??5@	?hү?&???hү?&??!?hү?&??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?,C??5@?ŏ1w??A????ҽ5@YjM??S??rEagerKernelExecute 0*	?????S@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh??|?5??!?d?hNC@)ݵ?|г??1?9kP<m@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??_?L??! ?-?9;@)r??????1נx??*7@:Preprocessing2U
Iterator::Model::ParallelMapV2M??St$??!?m?]Ɣ-@)M??St$??1?m?]Ɣ-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu????!W??l?S@)???Q?~?1+ ?-?#@:Preprocessing2F
Iterator::Model?St$????!????M?5@)?g??s?u?1?K8??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n??r?!/c??a	@)/n??r?1/c??a	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!'?eL?:@)a??+ei?1'?eL?:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapT㥛? ??!??2?|?D@)????Mb`?1????A?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?hү?&??I???l?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ŏ1w???ŏ1w??!?ŏ1w??      ??!       "      ??!       *      ??!       2	????ҽ5@????ҽ5@!????ҽ5@:      ??!       B      ??!       J	jM??S??jM??S??!jM??S??R      ??!       Z	jM??S??jM??S??!jM??S??b      ??!       JCPU_ONLYY?hү?&??b q???l?X@Y      Y@qA???
@"?
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