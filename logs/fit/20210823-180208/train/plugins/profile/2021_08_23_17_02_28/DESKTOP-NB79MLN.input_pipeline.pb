	????8@????8@!????8@	??~?U????~?U??!??~?U??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:????8@z6?>W??A?;Nё?7@Y/?$???rEagerKernelExecute 0*	?????K@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???&??!???5fHA@)vq?-??1?_c?4=@:Preprocessing2U
Iterator::Model::ParallelMapV2??0?*??!?v	ܻ?5@)??0?*??1?v	ܻ?5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip㥛? ???!?b???P@)lxz?,C|?1?I?!?)@:Preprocessing2F
Iterator::Model/n????!4:??#D@@)?????w?1D?"q%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?j+??݃?!??b??1@)?g??s?u?1??<???#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n??r?!4:??#D @)/n??r?14:??#D @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!D?"q@)?????g?1D?"q@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<?R??! ?/?%4@)a2U0*?S?19(g޲?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??~?U??I?@?{??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	z6?>W??z6?>W??!z6?>W??      ??!       "      ??!       *      ??!       2	?;Nё?7@?;Nё?7@!?;Nё?7@:      ??!       B      ??!       J	/?$???/?$???!/?$???R      ??!       Z	/?$???/?$???!/?$???b      ??!       JCPU_ONLYY??~?U??b q?@?{??X@