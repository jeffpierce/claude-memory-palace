import { Type } from "@sinclair/typebox";

export const message = {
  name: "palace_message",
  description: "Inter-instance messaging with pubsub. Actions: send, get, mark_read, mark_unread, subscribe, unsubscribe. Types: handoff, status, question, fyi, context, event, message. to_instance 'all' for broadcast. priority 0-10.",
  method: "message",
  parameters: Type.Object({
    action: Type.String(),
    instance_id: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    from_instance: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    to_instance: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    content: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    message_type: Type.Optional(Type.String({ default: "message" })),
    subject: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    channel: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    priority: Type.Optional(Type.Integer({ default: 0 })),
    unread_only: Type.Optional(Type.Boolean({ default: true })),
    limit: Type.Optional(Type.Integer({ default: 50 })),
    message_id: Type.Optional(Type.Union([Type.Integer(), Type.Null()])),
  }),
};
