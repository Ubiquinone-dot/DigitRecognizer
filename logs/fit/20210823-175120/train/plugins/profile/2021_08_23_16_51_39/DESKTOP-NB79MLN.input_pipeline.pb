	}гY?I9@}гY?I9@!}гY?I9@	?g$?"???g$?"??!?g$?"??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:}гY?I9@b??4?8??A?s?9@Yio???T??rEagerKernelExecute 0*	33333s^@2U
Iterator::Model::ParallelMapV20L?
F%??!?C??:?@@)0L?
F%??1?C??:?@@:Preprocessing2F
Iterator::Model??e??a??!??֡?K@)?]K?=??1?l???5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?b?=y??!?b5!Q?3@)M?O???1???
??0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateΈ?????!9D??.@)/?$???1????=!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?@??ǘ??!?2)^ F@)"??u????1!Q??6>@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ǘ????!y?*ڙ@)??ǘ????1y?*ڙ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???_vOn?!??;?XM@)???_vOn?1??;?XM@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'???????!??Ƈݑ1@)Ǻ???f?1?@&?d@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?g$?"??I1?̻?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b??4?8??b??4?8??!b??4?8??      ??!       "      ??!       *      ??!       2	?s?9@?s?9@!?s?9@:      ??!       B      ??!       J	io???T??io???T??!io???T??R      ??!       Z	io???T??io???T??!io???T??b      ??!       JCPU_ONLYY?g$?"??b q1?̻?X@