/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.iceberg.spark.source.parquet.vectorized;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.NullCheckingForGet;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.Files;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.arrow.vectorized.ArrowReader;
import org.apache.iceberg.arrow.vectorized.ColumnVector;
import org.apache.iceberg.arrow.vectorized.ColumnarBatch;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.io.InputFile;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.spark.source.IcebergSourceBenchmark;
import org.apache.iceberg.types.Types;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.infra.Blackhole;

import static org.apache.iceberg.types.Types.NestedField.optional;
import static org.apache.iceberg.types.Types.NestedField.required;

/**
 * Benchmark to compare performance of reading Parquet data with a flat schema using vectorized Iceberg read path and
 * the built-in file source in Spark.
 * <p>
 * To run this benchmark for either spark-2 or spark-3:
 * <code>
 * ./gradlew :iceberg-spark:iceberg-spark[2|3]:jmh
 * -PjmhIncludeRegex=VectorizedReadFlatParquetDataBenchmark
 * -PjmhOutputPath=benchmark/results.txt
 * </code>
 */
public class VectorizedReadFlatParquetDataBenchmark extends IcebergSourceBenchmark {

    private static final String FILE_PATH = getEnvOrProperty("ICEBERG_BENCHMARK_FILE",
            "iceberg.benchmark.file");

    private static String getEnvOrProperty(String envVariable, String propertyName) {
        String resultForEnv = System.getenv(envVariable);

        if (resultForEnv != null) {
            return resultForEnv;
        }

        String property = System.getProperty(propertyName);
        return property != null ? property : "";
    }

    @Override
    protected Configuration initHadoopConf() {
        return null;
    }

    @Override
    protected Table initTable() {
        return null;
    }

    @State(Scope.Thread)
    public static class BenchmarkState {
        public InputFile file = Files.localInput(FILE_PATH);

        // One important detail is that the first column in parquet file should have the index 1 in schema
        public Schema readSchemaInt = new Schema(
                optional(1, "f0", Types.IntegerType.get())
        );

        public Schema readSchemaBigInt = new Schema(
                optional(2, "f1", Types.LongType.get())
        );

        public Schema readSchemaFloat = new Schema(
                optional(3, "f2", Types.FloatType.get())
        );

        public Schema readSchemaDouble = new Schema(
                optional(4, "f3", Types.DoubleType.get())
        );

        public Schema readSchemaVarchar = new Schema(
                optional(5, "f4", Types.StringType.get())
        );

        public Schema readSchemaDate = new Schema(
                optional(7, "f6", Types.DateType.get())
        );

        public Schema readSchemaTime = new Schema(
                optional(8, "f7", Types.TimeType.get())
        );

        public Schema readSchemaTimestamp = new Schema(
                optional(9, "f8", Types.TimestampType.withoutZone())
        );
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeInt(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaInt)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaInt,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeBigInt(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaBigInt)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaBigInt,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeFloat(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaFloat)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaFloat,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeDouble(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaDouble)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaDouble,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeDate(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaDate)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaDate,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeTime(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaTime)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaTime,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeTimestamp(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaTimestamp)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaTimestamp,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Threads(1)
    public void testColumnPerTimeVarChar(BenchmarkState state, Blackhole blackhole) throws IOException {
        Parquet.ReadBuilder readBuilder = Parquet.read(state.file)
                .project(state.readSchemaVarchar)
                .createBatchedReaderFunc(fileSchema -> ArrowReader.VectorizedCombinedScanIterator.buildReader(state.readSchemaVarchar,
                        fileSchema, /* setArrowValidityVector */ NullCheckingForGet.NULL_CHECKING_ENABLED));


        try (CloseableIterable<ColumnarBatch> batchReader =
                     readBuilder.build()) {

            for (ColumnarBatch batch : batchReader) {
                ColumnVector columnVector = batch.column(0);
                FieldVector fieldVector = columnVector.getFieldVector();

                // Use this to avoid constant folding by the JVM
                blackhole.consume(fieldVector);
            }
        }
    }
}
