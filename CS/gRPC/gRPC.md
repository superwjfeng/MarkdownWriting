---
Title: gRPC
Author: Weijian Feng 封伟健
Source: Official documentation for gRPC <https://grpc.io/docs/>, 官方文档中文翻译 <https://doc.oschina.net/grpc?t=56831>
---

# gRPC

gRPC的源码库 <https://github.com/grpc/grpc>

## *intro*

gRPC（Google Remote Procedure Call）是一种高性能、通用的开源远程过程调用（RPC）框架，由Google开发并开源。它基于HTTP/2协议和Protocol Buffers（protobuf）序列化协议，用于构建跨平台和跨语言的分布式应用程序

gRPC 提供了一种简单的方法来定义服务接口和消息类型（**基于服务，与具体哪种语言无关**），并自动生成支持多种编程语言的客户端和服务器端代码。它支持多种编程语言，如C++、Java、Python、Go、C#等，因此可以在不同的环境和语言之间轻松进行通信

以下是 gRPC 的一些关键特性：

1. 基于标准的 HTTP/2：gRPC 使用 HTTP/2 作为底层的传输协议，提供了双向流、流控制、头部压缩等特性。HTTP/2 的多路复用特性允许同时处理多个请求和响应，提高了网络利用率和性能
2. 高效的序列化：gRPC 使用 Protocol Buffers（protobuf）作为默认的序列化机制。protobuf 是一种轻量级、高效的二进制数据交换格式，具有更小的消息大小和更快的序列化和反序列化速度
3. 自动化代码生成：通过定义服务接口和消息类型的 .proto 文件，gRPC 可以自动生成用于客户端和服务器端的代码。这简化了开发过程，并确保在不同的语言之间保持一致性
4. 支持多种调用方式：gRPC 支持四种不同的调用方式，满足不同的需求：
   * Unary RPC：一次请求，一次响应的调用方式
   * Server Streaming RPC：一次请求，多次响应的调用方式
   * Client Streaming RPC：多次请求，一次响应的调用方式
   * Bidirectional Streaming RPC：多次请求，多次响应的调用方式
5. 支持拦截器和中间件：gRPC 允许开发者使用拦截器和中间件来添加自定义的逻辑和功能。这包括身份验证、日志记录、错误处理等，提供了更灵活的扩展性和可定制性
6. 支持流式处理：gRPC 支持流式数据传输，可以用于处理大量数据、实时数据流和实时通信场景
7. 支持服务发现和负载均衡：gRPC 提供了服务发现和负载均衡的功能，可以轻松部署和管理大规模的分布式系统

gRPC 在微服务架构和分布式系统中具有广泛的应用。它提供了高效的网络通信和跨语言支持，使得不同服务能够方便地进行通信和协作。无论是构建大规模的云原生应用程序、实现高性能的数据传输，还是构建实时流处理系统，gRPC 都是一个强大的工具

## *项目结构*

1. xxxx-api 模块

    定义 protobuf IDL语言并通过命令创建具体的代码，后续client server引入使用

    * message
    * service

2. xxxx-server模块

    * 实现api模块中定义的服务接口
    * 发布gRPC服务（创建服务端程序）

3. xxxx-client模块
   * 创建服务端stub（代理）
   * 基于stub的RPC调用



## *C++ API*

1. **定义服务接口和消息类型**：首先，你需要使用 Protocol Buffers（protobuf）语言定义你的服务接口和消息类型。创建一个 `.proto` 文件，定义你的服务接口和消息类型。例如，创建一个 `hello_service.proto` 文件，其中包含一个简单的问候服务的定义

   ```protobuf
   syntax = "proto3";
   
   package com.example;
   
   service HelloService {
       rpc SayHello (HelloRequest) returns (HelloResponse);
   }
   
   message HelloRequest {
       string name = 1;
   }
   
   message HelloResponse {
       string message = 1;
   }
   ```

2. **生成 C++ 代码**：用 protobuf 编译器将 `.proto` 文件编译为 C++ 代码。你可以从官方网站下载 protobuf 编译器。在命令行中执行以下命令来生成 C++ 代码。这将生成与服务接口和消息类型对应的 C++ 文件

   ```shell
   protoc --cpp_out=. hello_service.proto
   ```

3. **实现服务接口**：在你的 C++ 代码中实现服务接口。为了实现 `HelloService` 接口，你需要创建一个类，并继承自生成的 `HelloService::Service` 类。在类中，实现服务接口定义的方法

   ```cpp
   #include <iostream>
   #include <grpcpp/grpcpp.h>
   #include "hello_service.grpc.pb.h"
   
   using grpc::Server;
   using grpc::ServerBuilder;
   using grpc::ServerContext;
   using grpc::Status;
   using hello::HelloRequest;
   using hello::HelloResponse;
   using hello::HelloService;
   
   class HelloServiceImpl final : public HelloService::Service {
       Status SayHello(ServerContext* context, const HelloRequest* request, HelloResponse* response) override {
           std::string name = request->name();
           std::string message = "Hello, " + name + "!";
           response->set_message(message);
           return Status::OK;
       }
   };
   
   void RunServer() {
       std::string server_address("0.0.0.0:50051");
       HelloServiceImpl service;
   
       ServerBuilder builder;
       builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
       builder.RegisterService(&service);
   
       std::unique_ptr<Server> server(builder.BuildAndStart());
       std::cout << "Server listening on " << server_address << std::endl;
       server->Wait();
   }
   
   int main() {
       RunServer();
       return 0;
   }
   ```

4. **编译和运行服务器**：使用 C++ 编译器编译你的服务器代码，并运行生成的可执行文件。确保你的编译命令中包含了 gRPC 和 Protocol Buffers 的库和头文件路径。这将启动一个监听指定地址的 gRPC 服务器

5. 创建 gRPC 客户端：在你的 C++ 代码中创建一个 gRPC 客户端，以便与服务器进行通信。首先，你需要创建一个 `Channel` 对象，指定服务器的地址和端口号。然后，你可以使用生成的客户端 stub（例如 `HelloService::NewStub(channel)`）来调用服务方法

   ```cpp
   #include <iostream>
   #include <grpcpp/grpcpp.h>
   #include "hello_service.grpc.pb.h"
   
   using grpc::Channel;
   using grpc::ClientContext;
   using grpc::Status;
   using hello::HelloRequest;
   using hello::HelloResponse;
   using hello::HelloService;
   
   class HelloClient {
   public:
       HelloClient(std::shared_ptr<Channel> channel)
           : stub_(HelloService::NewStub(channel)) {}
   
       std::string SayHello(const std::string& name) {
           HelloRequest request;
           request.set_name(name);
           HelloResponse response;
   
           ClientContext context;
           Status status = stub_->SayHello(&context, request, &response);
   
           if (status.ok()) {
               return response.message();
           } else {
               return "RPC failed";
           }
       }
   
   private:
       std::unique_ptr<HelloService::Stub> stub_;
   };
   
   int main() {
       std::string server_address("localhost:50051");
       HelloClient client(grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));
       std::string name("Alice");
       std::string response = client.SayHello(name);
       std::cout << "Response: " << response << std::endl;
       return 0;
   }
   ```

6. 编译和运行客户端：使用 C++ 编译器编译你的客户端代码，并运行生成的可执行文件。确保你的编译命令中包含了 gRPC 和 Protocol Buffers 的库和头文件路径。这将连接到服务器并调用服务方法，然后打印响应消息

   ```shell
   g++ -std=c++11 hello_service.pb.cc hello_service.grpc.pb.cc hello_client.cpp -o hello_client -lgrpc++ -lgrpc -lprotobuf
   ./hello_client
   ```

## *Java API*

1. 定义服务接口和消息类型：首先，你需要使用 Protocol Buffers（protobuf）语言定义你的服务接口和消息类型。protobuf 是一种用于序列化结构化数据的语言，它可以生成对应语言的代码，以便在你的应用程序中使用

   * 创建一个 `.proto` 文件，定义你的服务接口和消息类型。例如，创建一个 `HelloService.proto` 文件，其中包含一个简单的问候服务的定义

     ```protobuf
     syntax = "proto3";
     
     package com.example;
     
     service HelloService {
         rpc sayHello (HelloRequest) returns (HelloResponse);
     }
     
     message HelloRequest {
         string name = 1;
     }
     
     message HelloResponse {
         string message = 1;
     }
     ```

2. 生成 Java 代码：使用 protobuf 编译器将 `.proto` 文件编译为 Java 代码。你可以从官方网站下载 protobuf 编译器。在命令行中执行以下命令来生成 Java 代码。这将生成与服务接口和消息类型对应的 Java 类

   ```shell
   protoc --java_out=. HelloService.proto
   ```

3. 实现服务接口：在你的 Java 代码中实现服务接口。为了实现 `HelloService` 接口，你需要创建一个类，并继承自生成的 `HelloServiceGrpc.HelloServiceImplBase` 类。在类中，实现服务接口定义的方法

   ```java
   package com.example;
   
   import io.grpc.stub.StreamObserver;
   
   public class HelloServiceImpl extends HelloServiceGrpc.HelloServiceImplBase {
   
       @Override
       public void sayHello(HelloRequest request, StreamObserver<HelloResponse> responseObserver) {
           String name = request.getName();
           String message = "Hello, " + name + "!";
   
           HelloResponse response = HelloResponse.newBuilder()
                   .setMessage(message)
                   .build();
   
           responseObserver.onNext(response);
           responseObserver.onCompleted();
       }
   }
   ```

4. 启动 gRPC 服务器：创建一个 gRPC 服务器，将你的服务实现添加到服务器中，并启动服务器以监听客户端请求

   ```java
   package com.example;
   
   import io.grpc.Server;
   import io.grpc.ServerBuilder;
   
   import java.io.IOException;
   
   public class HelloServer {
   
       public static void main(String[] args) throws IOException, InterruptedException {
           Server server = ServerBuilder.forPort(8080)
                   .addService(new HelloServiceImpl())
                   .build();
   
           server.start();
   
           System.out.println("Server started");
   
           server.awaitTermination();
       }
   }
   ```

5. 创建 gRPC 客户端：在你的 Java 代码中创建一个 gRPC 客户端，以便与服务器进行通信。首先，你需要创建一个 `ManagedChannel` 对象，指定服务器的主机名和端口号。然后，你可以使用生成的客户端 stub（例如 `HelloServiceGrpc.HelloServiceBlockingStub`）来调用服务方法

   ```java
   package com.example;
   
   import io.grpc.ManagedChannel;
   import io.grpc.ManagedChannelBuilder;
   
   public class HelloClient {
   
       public static void main(String[] args) {
           ManagedChannel channel = ManagedChannelBuilder.forAddress("localhost", 8080)
                   .usePlaintext()
                   .build();
   
           HelloServiceGrpc.HelloServiceBlockingStub stub = HelloServiceGrpc.newBlockingStub(channel);
   
           HelloRequest request = HelloRequest.newBuilder()
                   .setName("Alice")
                   .build();
   
           HelloResponse response = stub.sayHello(request);
   
           System.out.println(response.getMessage());
   
           channel.shutdown();
       }
   }
   ```









```protobuf
// 后续protobuf生成的java代码一个源文件还是多个源文件xx.java
option java_multiple_files = false;
// 指定protobuf生成的类放置在哪个包中
option java_package = "com. suns";
// 指定的protobuf生成的外部类的名字（管理内部类【内部类オ是真正开发使用】）
option java_outer_classname = "UserServoe";
```







```protobuf
service HelloService{
	rpc hello (HelloRequest) returns (HelloResponse) {}
}
＃里面是可以定义多个服务方法
```

